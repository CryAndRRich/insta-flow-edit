from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from .helper import calculate_shift, lr_hump_tail_beta

__SAMPLER__ = {}

def register_sampler(name: str) -> callable:
    """Decorator to register a new sampler class"""
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_flux_sampler(name: str, **kwargs) -> object:
    """Factory method to get a sampler instance by name"""
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist.")
    return __SAMPLER__[name](**kwargs)


class FluxBase:
    def __init__(self, 
                 model_key: str = "black-forest-labs/FLUX.1-dev", 
                 device: str = "cuda", 
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False,
                 quantize_4bit: bool = True) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading FLUX Base from {model_key} (Offload: {offload}, 4-bit: {quantize_4bit})...")

        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype
            )
            
            transformer_device_map = "auto" if not offload else None
            
            # Load Transformer (4-bit)
            transformer = FluxTransformer2DModel.from_pretrained(
                model_key,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
                device_map=transformer_device_map 
            )
            
            # Load T5 Encoder (4-bit)
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_key,
                subfolder="text_encoder_2",
                quantization_config=quant_config,
                torch_dtype=dtype,
                device_map=transformer_device_map
            )
            
            # Initialize Pipeline
            pipe = FluxPipeline.from_pretrained(
                model_key,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=dtype
            )
        else:
            # Standard Load
            pipe = FluxPipeline.from_pretrained(model_key, torch_dtype=dtype)

        # Enable VAE Slicing
        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        # Device Management
        if self.offload:
            pipe.enable_model_cpu_offload() 
        else:
            if not quantize_4bit:
                pipe.to(device)
            else:
                pipe.text_encoder.to(device)
                pipe.vae.to(device)

        self.scheduler = pipe.scheduler

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2

        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_2.eval()
        self.text_encoder_2.requires_grad_(False)
        
        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = pipe.vae_scale_factor
        
        del pipe

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device if not self.offload else "cuda"
        
        # CLIP Encoding
        text_clip_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        pooled_prompt_embeds = self.text_encoder(text_clip_ids.to(device), output_hidden_states=False).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype, device=device)

        # T5
        text_t5_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512, 
            truncation=True,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder_2(text_t5_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)

        # RoPE Text IDs
        txt_ids = torch.zeros(prompt_embeds.shape[0], prompt_embeds.shape[1], 3).to(device=device, dtype=self.dtype)

        return prompt_embeds, pooled_prompt_embeds, txt_ids

    def prepare_latents(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)
        
        # VAE Encode
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        h_lat, w_lat = z.shape[2], z.shape[3]
        
        # Pack Latents (B, C, H, W) -> (B, L, C_packed)
        batch_size, num_channels, height, width = z.shape
        z_packed = self._pack_latents(z, batch_size, num_channels, height, width)
        
        # Create Latent Image IDs (RoPE)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, self.dtype)

        return z_packed, latent_image_ids, h_lat, w_lat

    def _pack_latents(self, 
                      latents: torch.Tensor, 
                      batch_size: int, 
                      num_channels_latents: int, 
                      height: int, 
                      width: int) -> torch.Tensor:
        # FLUX specific packing: reshape to (B, H/2, W/2, C*4) -> (B, L, C*4)
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _unpack_latents(self, 
                        latents: torch.Tensor, 
                        height: int, 
                        width: int) -> torch.Tensor:
        batch_size = latents.shape[0]
        num_channels_latents = (latents.shape[2] // 4) 
        
        latents = latents.view(batch_size, height // 2, width // 2, num_channels_latents, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, num_channels_latents, height, width)
        return latents

    def _prepare_latent_image_ids(self, 
                                  batch_size: int, 
                                  height: int, 
                                  width: int, 
                                  device: str, 
                                  dtype: torch.dtype) -> torch.Tensor:
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(batch_size, (height // 2) * (width // 2), 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def decode(self, 
               z_packed: torch.Tensor, 
               h_lat: int, 
               w_lat: int) -> torch.Tensor:
        """Unpacks latents and decodes them back to image space"""
        z = self._unpack_latents(z_packed, h_lat, w_lat, self.vae_scale_factor)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, 
                       z: torch.Tensor, 
                       t: torch.Tensor, 
                       prompt_emb: torch.Tensor, 
                       pooled_emb: torch.Tensor,
                       txt_ids: torch.Tensor,
                       img_ids: torch.Tensor,
                       guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(t.shape) == 0: 
            t = t.expand(z.shape[0])
        t = t.to(dtype=self.dtype)

        if self.offload:
            z = z.to("cuda")
            t = t.to("cuda")
            prompt_emb = prompt_emb.to("cuda")
            pooled_emb = pooled_emb.to("cuda")
            txt_ids = txt_ids.to("cuda")
            img_ids = img_ids.to("cuda")
            if guidance is not None:
                guidance = guidance.to("cuda")

        v = self.transformer(
            hidden_states=z,
            timestep=t, 
            guidance=guidance,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False
        )[0]
        
        return v
    
    def prepare_embed(self, 
                      prompt: str, 
                      embs: Optional[Tuple]) -> Tuple:
        if embs is None:
            return self.encode_prompt(prompt)
        device = self.device if not self.offload else "cuda"
        return tuple(e.to(device) for e in embs)


@register_sampler(name="flowedit")
class FluxFlowEdit(FluxBase):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               NFE: int = 28, 
               n_start: int = 0, 
               tar_cfg_scale: float = 5.5,
               src_cfg_scale: float = 1.5,
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of FlowEdit for FLUX.1
        
        Parameters:
            src_img: Input source image tensor [B, C, H, W]
            src_prompt: Source prompt string
            tgt_prompt: Target prompt string
            NFE: Number of function evaluations
            n_start: Number of initial steps to skip editing
            tar_cfg_scale: Classifier-free guidance scale for target prompt
            src_cfg_scale: Classifier-free guidance scale for source prompt
            src_prompt_emb: Optional precomputed embeddings for source prompt
            tgt_prompt_emb: Optional precomputed embeddings for target prompt
            
        Returns:
            img: Edited output image tensor [B, C, H, W]
        """
        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
        
        # Encode Source and Pack
        with torch.no_grad():
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)
        
        # Initialize editing latent
        x_t = z_src.clone()

        # Setup Scheduler
        sigmas = np.linspace(1.0, 1 / NFE, NFE)
        
        # Calculate Shift based on packed latent sequence length
        mu = calculate_shift(
            z_src.shape[1], # Sequence length
            self.scheduler.config.base_image_seq_len, 
            self.scheduler.config.max_image_seq_len, 
            self.scheduler.config.base_shift, 
            self.scheduler.config.max_shift
        )
        
        # Retrieve timesteps applying the Shift
        timesteps, _ = retrieve_timesteps(
            self.scheduler, 
            NFE, 
            self.device, 
            timesteps=None, 
            sigmas=sigmas, 
            mu=mu
        )
        
        device = self.device if not self.offload else "cuda"
        src_guidance = torch.tensor([src_cfg_scale], device=device).expand(z_src.shape[0])
        tar_guidance = torch.tensor([tar_cfg_scale], device=device).expand(z_src.shape[0])

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="FLUX FlowEdit")
        for i, t in pbar:
            if i < n_start: 
                continue
            
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
            dt = t_next - t_curr
            t_tensor = t_curr.view(1)

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                v_tar = self.predict_vector(
                    pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=tar_guidance
                )
                
                v_src = self.predict_vector(
                    qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=src_guidance
                )

            x_t = x_t + dt * (v_tar - v_src)

        with torch.no_grad():
            img = self.decode(x_t, h_lat, w_lat)
        return img


@register_sampler(name="flowalign")
class FluxFlowAlign(FluxBase):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               NFE: int = 28, 
               n_start: int = 0, 
               cfg_scale: float = 3.5,
               zeta: float = 0.01,
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of FlowAlign for FLUX.1
        
        Parameters:
            src_img: Input source image tensor (B, C, H, W)
            src_prompt: Source prompt string
            tgt_prompt: Target prompt string
            NFE: Number of function evaluations
            n_start: Number of initial steps to skip editing
            cfg_scale: Classifier-free guidance scale for target prompt
            zeta: Regularization strength
            src_prompt_emb: Optional precomputed embeddings for source prompt
            tgt_prompt_emb: Optional precomputed embeddings for target prompt
        
        Returns:
            img: Edited output image tensor (B, C, H, W)
        """
        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        # Encode Source
        with torch.no_grad():
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)
        
        # Initialize editing Latent
        x_t = z_src.clone()

        # Setup Scheduler
        sigmas = np.linspace(1.0, 1 / NFE, NFE)
        mu = calculate_shift(
            z_src.shape[1], 
            self.scheduler.config.base_image_seq_len, 
            self.scheduler.config.max_image_seq_len, 
            self.scheduler.config.base_shift, 
            self.scheduler.config.max_shift
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, 
            NFE, 
            self.device, 
            timesteps=None, 
            sigmas=sigmas, 
            mu=mu
        )
        
        device = self.device if not self.offload else "cuda"
        tar_guidance = torch.tensor([cfg_scale], device=device).expand(z_src.shape[0])
        src_guidance = torch.tensor([1.0], device=device).expand(z_src.shape[0])

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="FLUX FlowAlign")
        for i, t in pbar:
            if i < n_start: 
                continue
            
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
            dt = t_next - t_curr
            t_tensor = t_curr.view(1)

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                v_tar = self.predict_vector(
                    pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=tar_guidance
                )
                
                v_src = self.predict_vector(
                    qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=src_guidance
                )

            # Regularization Term
            # Reg = (qt - pt) + sigma * (v_tar - v_src)
            reg_term = (qt - pt) + t_curr * (v_tar - v_src)

            # Update with Regularization
            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t, h_lat, w_lat)
        return img
    

@register_sampler(name="dvrf")
class FluxDeltaVelFlow(FluxBase):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               steps: int = 50, # DVRF uses "steps" for optimization iterations
               eta: float = 1.0, 
               lr_max: float = 0.04,
               tar_cfg_scale: float = 5.5,
               src_cfg_scale: float = 1.0,
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of Delta Velocity Rectified Flow (DVRF) for FLUX
        
        Parameters:
            src_img: Input source image tensor [B, C, H, W]
            src_prompt: Source prompt string
            tgt_prompt: Target prompt string
            steps: Number of optimization steps
            eta: Shift coefficient strength
            lr_max: Maximum learning rate
            tar_cfg_scale: Guidance for target
            src_cfg_scale: Guidance for source
            src_prompt_emb: Optional precomputed embeddings for source prompt
            tgt_prompt_emb: Optional precomputed embeddings for target prompt
        
        Returns:
            img: Edited output image tensor
        """
        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        # Encode Source Latents & Pack
        with torch.no_grad():
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        z_tgt = z_src.clone().detach().requires_grad_(True)
        
        # Initialize Optimizer
        optimizer = torch.optim.SGD([z_tgt], lr=lr_max)

        # Setup Timesteps
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            z_src.shape[1], 
            self.scheduler.config.base_image_seq_len, 
            self.scheduler.config.max_image_seq_len, 
            self.scheduler.config.base_shift, 
            self.scheduler.config.max_shift
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, 
            steps, 
            self.device, 
            timesteps=None, 
            sigmas=sigmas, 
            mu=mu
        )
        
        # Prepare Guidance Tensors
        device = self.device if not self.offload else "cuda"
        tar_guidance = torch.tensor([tar_cfg_scale], device=device).expand(z_src.shape[0])
        src_guidance = torch.tensor([src_cfg_scale], device=device).expand(z_src.shape[0])

        # Learning Rate Schedule Params
        alpha_T_steps = (timesteps[steps - 2].item() / 1000 - timesteps[steps - 1].item() / 1000) / 1.6
        # Fallback if calculation fails
        alpha_max_dynamic = alpha_T_steps / 1.6 if alpha_T_steps > 0 else lr_max
        beta_tail = alpha_T_steps / 4 if alpha_T_steps > 0 else lr_max / 4

        # Optimization Loop
        pbar = tqdm(enumerate(timesteps), total=steps, desc="FLUX DVRF")
        
        for i, t in pbar:
            t_curr = t
            
            # Update Learning Rate
            current_lr = 2.2 * lr_hump_tail_beta(i+1, steps + 28, alpha_max_dynamic, beta_tail, a=10, b=8)
            optimizer.param_groups[0]["lr"] = current_lr
            
            # Shift coefficient
            eta_i = eta * i / steps
            
            eps = torch.randn_like(z_src)
            
            # Forward Process (Construct noisy latents)
            # Source Branch: qt = (1 - t) * z_src + t * eps
            qt = (1 - t_curr) * z_src + t_curr * eps
            
            # Target Branch (with Shift): pt = (1 - t) * z_tgt + t * eps + eta_i * t * (z_tgt - z_src)
            z_tgt_detached = z_tgt.detach()
            # Shift term calculation
            shift_term = eta_i * t_curr * (z_tgt_detached - z_src)
            pt = (1 - t_curr) * z_tgt_detached + t_curr * eps + shift_term

            # Predict Velocities 
            t_tensor = t_curr.view(1)
            
            with torch.inference_mode():
                v_tar = self.predict_vector(
                    pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=tar_guidance
                )
                
                v_src = self.predict_vector(
                    qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=src_guidance
                )
            
            # Calculate Gradient Approximation
            # grad = (v_tar - v_src) + (1 - eta_i) * (z_tgt - z_src)
            delta_v = v_tar - v_src
            geometric_term = (1 - eta_i) * (z_tgt_detached - z_src)
            grad = delta_v + geometric_term
            
            # Manual Backward Pass
            # loss = sum(z_tgt * grad) -> d(loss)/d(z_tgt) = grad
            loss = (z_tgt * grad.detach()).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"lr": f"{current_lr:.5f}", "eta": f"{eta_i:.2f}"})

        with torch.no_grad():
            img = self.decode(z_tgt.detach(), h_lat, w_lat)
        return img