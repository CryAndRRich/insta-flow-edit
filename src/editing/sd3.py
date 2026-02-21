from typing import List, Tuple, Optional
import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusion3Pipeline
from .helper import lr_hump_tail_beta

__SAMPLER__ = {}

def register_sampler(name: str) -> callable:
    """Decorator to register a new sampler class"""
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sd3_sampler(name: str, **kwargs) -> object:
    """Factory method to get a sampler instance by name"""
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist")
    return __SAMPLER__[name](**kwargs)


class StableDiffusion3Base:
    def __init__(self, 
                 model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers", 
                 device: str = "cuda", 
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading SD3 Base from {model_key} (Offload: {offload})...")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe.vae, "enable_slicing"):
            try:
                pipe.vae.enable_slicing()
            except:
                pass

        if self.offload:
            pipe.enable_model_cpu_offload()

        self.scheduler = pipe.scheduler
        
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_1.eval()
        self.text_enc_1.requires_grad_(False)
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_2.eval()
        self.text_enc_2.requires_grad_(False)
        self.text_enc_3 = pipe.text_encoder_3
        self.text_enc_3.eval()
        self.text_enc_3.requires_grad_(False)

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        
        if not self.offload:
            self.text_enc_1.to(device)
            self.text_enc_2.to(device)
            self.text_enc_3.to(device)
            self.vae.to(device)
            self.transformer.to(device)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        del pipe

    def encode_prompt(self, prompt: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device if not self.offload else "cuda"

        # T5
        text_t5_ids = self.tokenizer_3(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True,
            add_special_tokens=True, 
            return_tensors="pt"
        ).input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(device))[0].to(dtype=self.dtype, device=device)

        # CLIP
        text_clip1_ids = self.tokenizer_1(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        text_clip2_ids = self.tokenizer_2(
            prompt, 
            padding="max_length", 
            max_length=77, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        # Merge Embeddings
        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1]))
        
        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)
        
        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, 
                       z: torch.Tensor, 
                       t: torch.Tensor, 
                       prompt_emb: torch.Tensor, 
                       pooled_emb: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0: 
            t = t.expand(z.shape[0])
        t = t.to(dtype=self.dtype)

        if self.offload:
            z = z.to("cuda")
            t = t.to("cuda")
            prompt_emb = prompt_emb.to("cuda")
            pooled_emb = pooled_emb.to("cuda")

        v = self.transformer(
            hidden_states=z, 
            timestep=t, 
            pooled_projections=pooled_emb, 
            encoder_hidden_states=prompt_emb, 
            return_dict=False
        )[0]
        return v

    def prepare_embed(self, 
                      prompt: str, 
                      embs: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if embs is None:
            return self.encode_prompt([prompt])
        
        device = self.device if not self.offload else "cuda"
        return embs[0].to(device), embs[1].to(device)


@register_sampler(name="flowedit")
class SD3FlowEdit(StableDiffusion3Base):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               neg_prompt: str = "", 
               NFE: int = 50, 
               n_start: int = 0, 
               tar_cfg_scale: float = 13.5, 
               src_cfg_scale: float = 3.5,
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None, 
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of FlowEdit for Stable Diffusion 3

        Parameters:
            src_img: Input source image tensor [1, 3, H, W]
            src_prompt: Description of the source image
            tgt_prompt: Description of the desired target image
            neg_prompt: Negative prompt for CFG
            NFE: Number of Function Evaluations
            n_start: Step index to start editing (skipping initial noise levels)
            tar_cfg_scale: Guidance scale for the target branch
            src_cfg_scale: Guidance scale for the source branch
            src_prompt_emb: Pre-computed source prompt embeddings 
            tgt_prompt_emb: Pre-computed target prompt embeddings
            neg_prompt_emb: Pre-computed negative prompt embeddings
        
        Returns:
            img: Edited output image tensor [1, 3, H, W]
        """

        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)

        # Encode Source Image
        with torch.no_grad():
            z_src = self.encode(src_img)
        
        # Initialize editing latent (x_t) same as source latent (z_src)
        x_t = z_src.clone()

        # Setup Scheduler
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc="SD3 FlowEdit")
        for i, t in enumerate(pbar):
            # Skip initial steps if n_start > 0
            if i < n_start: 
                continue

            sigma = sigmas[i]
            # Handle last step case
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma 

            eps = torch.randn_like(z_src)
            qt = (1 - sigma) * z_src + sigma * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                v_tar_cond = self.predict_vector(pt, t, tgt_emb, tgt_pool)
                v_tar_neg = self.predict_vector(pt, t, neg_emb, neg_pool)
                v_tar = v_tar_neg + tar_cfg_scale * (v_tar_cond - v_tar_neg)
                
                v_src_cond = self.predict_vector(qt, t, src_emb, src_pool)
                v_src_neg = self.predict_vector(qt, t, neg_emb, neg_pool)
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)

            # Euler update
            x_t = x_t + dt * (v_tar - v_src)

        with torch.no_grad():
            img = self.decode(x_t)
        return img 


@register_sampler(name="flowalign")
class SD3FlowAlign(StableDiffusion3Base):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               neg_prompt: str = "", 
               NFE: int = 50, 
               n_start: int = 0, 
               cfg_scale: float = 7.0, 
               zeta: float = 0.01,     
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None, 
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of FlowAlign for Stable Diffusion 3

        Parameters:
            src_img: Input source image tensor [1, 3, H, W], normalized [-1, 1]
            src_prompt: Description of the source image
            tgt_prompt: Description of the desired target image
            neg_prompt: Negative prompt for CFG
            NFE: Number of Function Evaluations
            n_start: Step index to start editing
            cfg_scale: Guidance scale for the target branch only
            zeta: Regularization strength (controls background preservation)
            src_prompt_emb: Pre-computed source prompt embeddings 
            tgt_prompt_emb: Pre-computed target prompt embeddings
            neg_prompt_emb: Pre-computed negative prompt embeddings
        
        Returns:
            img: Edited output image tensor [1, 3, H, W], normalized [-1, 1]
        """

        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)

        # Encode Source Image
        with torch.no_grad():
            z_src = self.encode(src_img)
        
        # Initialize editing latent
        x_t = z_src.clone()

        # Setup Scheduler
        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc="SD3 FlowAlign")
        for i, t in enumerate(pbar):
            if i < n_start: 
                continue

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma 

            eps = torch.randn_like(z_src)
            qt = (1 - sigma) * z_src + sigma * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                v_tar_cond = self.predict_vector(pt, t, tgt_emb, tgt_pool)
                v_tar_neg = self.predict_vector(pt, t, neg_emb, neg_pool)
                v_tar = v_tar_neg + cfg_scale * (v_tar_cond - v_tar_neg)
                
                v_src = self.predict_vector(qt, t, src_emb, src_pool)

            # Formula: Reg = (qt - pt) + sigma * (v_tar - v_src)
            # Represents the difference between Tweedie estimates E[q0] and E[p0]
            reg_term = (qt - pt) + sigma * (v_tar - v_src)

            # Update with Regularization
            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t)
        return img


@register_sampler(name="dvrf")
class SD3DeltaVelFlow(StableDiffusion3Base):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               steps: int = 50,  
               eta: float = 1.0, 
               lr_max: float = 0.04, 
               tar_cfg_scale: float = 16.5,
               src_cfg_scale: float = 6.0,
               src_prompt_emb: Optional[Tuple] = None, 
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        Implementation of Delta Velocity Rectified Flow (DVRF) for Stable Diffusion 3
        
        Parameters:
            src_img: Input source image tensor [1, 3, H, W]
            src_prompt: Source prompt string
            tgt_prompt: Target prompt string
            steps: Number of optimization steps
            eta: Shift coefficient strength
            lr_max: Maximum learning rate
            tar_cfg_scale: Guidance for target
            src_cfg_scale: Guidance for source
            src_prompt_emb: Pre-computed source prompt embeddings
            tgt_prompt_emb: Pre-computed target prompt embeddings
        
        Returns:
            img: Edited output image tensor [1, 3, H, W]
        """
        
        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb) 
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        # Encode Source Image
        with torch.no_grad():
            z_src = self.encode(src_img)

        z_tgt = z_src.clone().detach().requires_grad_(True)

        # Initialize Optimizer
        optimizer = torch.optim.SGD([z_tgt], lr=lr_max)

        # Setup Scheduler
        self.scheduler.set_timesteps(steps, device=self.device)
        timesteps = self.scheduler.timesteps 

        # Calculate dynamic LR params
        t_start = timesteps[0].item() / 1000.0
        t_end = timesteps[-1].item() / 1000.0 
        alpha_T_steps = (t_start - t_end) / steps / 1.6 # Approximate
        alpha_max_dynamic = alpha_T_steps / 1.6 if alpha_T_steps > 0 else lr_max
        beta_tail = alpha_T_steps / 4 if alpha_T_steps > 0 else lr_max / 4

        pbar = tqdm(enumerate(timesteps), total=steps, desc="SD3 DVRF")
        
        for i, t in pbar:
            t_curr = t 
            t_scalar = t.item() / 1000.0 
            
            # Update Learning Rate
            current_lr = 2.2 * lr_hump_tail_beta(i + 1, steps + 28, alpha_max_dynamic, beta_tail, a=10, b=8)
            optimizer.param_groups[0]["lr"] = current_lr
            
            # Shift coefficient (eta_i)
            eta_i = eta * i / steps
            
            eps = torch.randn_like(z_src)
            
            # Forward Process (Construct noisy latents)
            # Source Branch: qt = (1 - t) * z_src + t * eps
            qt = (1 - t_scalar) * z_src + t_scalar * eps
            
            # Target Branch (with Shift): pt = (1 - t) * z_tgt + t * eps + eta_i * t * (z_tgt - z_src)
            z_tgt_detached = z_tgt.detach()
            shift_term = eta_i * t_scalar * (z_tgt_detached - z_src)
            pt = (1 - t_scalar) * z_tgt_detached + t_scalar * eps + shift_term

            # Predict Velocities
            with torch.inference_mode():
                v_tar_cond = self.predict_vector(pt, t_curr, tgt_emb, tgt_pool)
                if i == 0:
                    neg_emb, neg_pool = self.encode_prompt([""])
                
                v_tar_neg = self.predict_vector(pt, t_curr, neg_emb, neg_pool)
                v_tar = v_tar_neg + tar_cfg_scale * (v_tar_cond - v_tar_neg)
                
                v_src_cond = self.predict_vector(qt, t_curr, src_emb, src_pool)
                v_src_neg = self.predict_vector(qt, t_curr, neg_emb, neg_pool)
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)

            # Calculate Gradient Approximation
            delta_v = v_tar - v_src
            geometric_term = (1 - eta_i) * (z_tgt_detached - z_src)
            grad = delta_v + geometric_term

            # Manual Backward Pass
            loss = (z_tgt * grad.detach()).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"lr": f"{current_lr:.5f}", "eta": f"{eta_i:.2f}"})

        with torch.no_grad():
            img = self.decode(z_tgt.detach())
            
        return img