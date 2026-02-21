from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline
from .helper import lr_hump_tail_beta

__SAMPLER__ = {}

def register_sampler(name: str) -> callable:
    """Decorator to register a new sampler class"""
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_instaflow_sampler(name: str, **kwargs) -> object:
    """Factory method to get a sampler instance by name"""
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist.")
    return __SAMPLER__[name](**kwargs)


class InstaFlowBase:
    def __init__(self, 
                 model_key: str = "XCLiu/2_rectified_flow_from_sd_1_5", 
                 device: str = "cuda", 
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading InstaFlow Base from {model_key} (Offload: {offload})...")
        
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        pipe.safety_checker = None

        if self.offload:
            pipe.enable_model_cpu_offload()
        
        self.scheduler = pipe.scheduler

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        self.unet = pipe.unet
        self.unet.eval()
        self.unet.requires_grad_(False)
        
        if not self.offload:
            self.vae.to(device)
            self.text_encoder.to(device)
            self.unet.to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        del pipe

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        return prompt_embeds.to(dtype=self.dtype, device=device)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)
        
        z = self.vae.encode(image).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self, 
                       z: torch.Tensor, 
                       t: torch.Tensor, 
                       prompt_embeds: torch.Tensor) -> torch.Tensor:
        """
        Predicts velocity from the U-Net. InstaFlow is a Rectified Flow model finetuned from SD1.5
        """
        # InstaFlow uses timestep range [0, 1000] for input
        timestep = t * self.scheduler.config.num_train_timesteps
        
        timestep = timestep.to(z.device).to(z.dtype)
        if len(timestep.shape) == 0:
            timestep = timestep.expand(z.shape[0])

        if self.offload:
            z = z.to("cuda")
            timestep = timestep.to("cuda")
            prompt_embeds = prompt_embeds.to("cuda")

        # Predict Noise
        noise_pred = self.unet(
            z,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # Convert Noise to Velocity (v = alpha * noise - beta * x)
        # Note: InstaFlow uses standard schedulers, we need to fetch alpha/beta
        step_index = (t * (self.scheduler.config.num_train_timesteps - 1)).long()
        if isinstance(step_index, torch.Tensor):
            step_index = step_index.item() # Assuming scalar for simplicity or handle batch
            
        alpha_prod_t = self.scheduler.alphas_cumprod[step_index]
        alpha_prod_t = alpha_prod_t.to(z.device).to(z.dtype)
        beta_prod_t = 1 - alpha_prod_t
        
        alpha_sqrt = alpha_prod_t ** 0.5
        beta_sqrt = beta_prod_t ** 0.5
        
        v = alpha_sqrt * noise_pred - beta_sqrt * z
        
        return v
    
    def prepare_embed(self, 
                      prompt: str, 
                      neg_prompt: str = "",
                      embs: Optional[torch.Tensor] = None,
                      neg_embs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if embs is None:
            embs = self.encode_prompt(prompt)
        
        if neg_embs is None:
            neg_embs = self.encode_prompt(neg_prompt)
            
        device = self.device if not self.offload else "cuda"
        return embs.to(device), neg_embs.to(device)


@register_sampler(name="flowedit")
class InstaFlowEdit(InstaFlowBase):
    """
    Implementation of FlowEdit for InstaFlow.
    Uses double CFG (Source and Target) without regularization.
    """
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               neg_prompt: str = "", 
               NFE: int = 25, 
               n_start: int = 0, 
               tar_cfg_scale: float = 16.5,
               src_cfg_scale: float = 1.5,
               src_prompt_emb: Optional[torch.Tensor] = None, 
               tgt_prompt_emb: Optional[torch.Tensor] = None, 
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of FlowEdit for InstaFlow.

        Parameters:
            src_img: Input source image tensor [1, 3, H, W]
            src_prompt: Description of the source image
            tgt_prompt: Description of the desired target image
            neg_prompt: Negative prompt
            NFE: Number of Function Evaluations
            n_start: Step index to start editing
            tar_cfg_scale: Guidance scale for target
            src_cfg_scale: Guidance scale for source
            src_prompt_emb: Optional precomputed source prompt embeddings
            tgt_prompt_emb: Optional precomputed target prompt embeddings
            neg_prompt_emb: Optional precomputed negative prompt embeddings
        
        Returns:
            img: Edited output image tensor [1, 3, H, W]
        """

        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb) 
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            
            # Combine for batch processing: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
            combined_embeds = torch.cat([src_neg, src_emb, tgt_neg, tgt_emb], dim=0)

        # Encode Source Image
        with torch.no_grad():
            z_src = self.encode(src_img)
        
        # Initialize editing latent
        x_t = z_src.clone()

        # Setup Scheduler (Linear 1.0 -> 0.0)
        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)
        
        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow FlowEdit")
        for i, t in pbar:
            if i < n_start: 
                continue
            
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = abs(t_next - t_curr) # InstaFlow uses positive dt logic in update usually

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                latent_input = torch.cat([qt, qt, pt, pt])
                
                # Predict Velocity
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                
                # Split chunks
                v_src_uncond, v_src_text, v_tar_uncond, v_tar_text = v_all.chunk(4)
                
                # Apply CFG
                v_src = v_src_uncond + src_cfg_scale * (v_src_text - v_src_uncond)
                v_tar = v_tar_uncond + tar_cfg_scale * (v_tar_text - v_tar_uncond)

            # Euler Update
            x_t = x_t + dt * (v_tar - v_src)

        with torch.no_grad():
            img = self.decode(x_t)
        return img


@register_sampler(name="flowalign")
class InstaFlowAlign(InstaFlowBase):
    """
    Implementation of FlowAlign for InstaFlow.
    Uses Single CFG (Target) + Regularization.
    """
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               neg_prompt: str = "", 
               NFE: int = 25, 
               n_start: int = 0, 
               cfg_scale: float = 16.5, 
               zeta: float = 0.01,
               src_prompt_emb: Optional[torch.Tensor] = None, 
               tgt_prompt_emb: Optional[torch.Tensor] = None, 
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of FlowAlign for InstaFlow

        Parameters:
            src_img: Input source image tensor [1, 3, H, W]
            src_prompt: Description of the source image
            tgt_prompt: Description of the desired target image
            neg_prompt: Negative prompt
            NFE: Number of Function Evaluations
            n_start: Step index to start editing
            cfg_scale: Guidance scale for target
            zeta: Regularization strength
            src_prompt_emb: Optional precomputed source prompt embeddings
            tgt_prompt_emb: Optional precomputed target prompt embeddings
            neg_prompt_emb: Optional precomputed negative prompt embeddings

        Returns:
            img: Edited output image tensor [1, 3, H, W]
        """

        # 1. Prepare Embeddings
        with torch.no_grad():
            src_emb, _ = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb) 
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            
            combined_embeds = torch.cat([src_emb, tgt_neg, tgt_emb], dim=0)

        with torch.no_grad():
            z_src = self.encode(src_img)
        
        x_t = z_src.clone()

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)
        
        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow FlowAlign")
        for i, t in pbar:
            if i < n_start: 
                continue
            
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = abs(t_next - t_curr)

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps 
            pt = x_t + qt - z_src 

            with torch.no_grad():
                latent_input = torch.cat([qt, pt, pt])
                
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_src, v_tar_uncond, v_tar_text = v_all.chunk(3)
                
                v_tar = v_tar_uncond + cfg_scale * (v_tar_text - v_tar_uncond)
                
            # Regularization Term
            # Reg = (qt - pt) + t * (v_tar - v_src)
            reg_term = (qt - pt) + t_curr * (v_tar - v_src)

            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t)
        return img
    

@register_sampler(name="dvrf")
class InstaDeltaVelFlow(InstaFlowBase):
    def sample(self, 
               src_img: torch.Tensor, 
               src_prompt: str, 
               tgt_prompt: str, 
               neg_prompt: str = "",
               steps: int = 50, 
               eta: float = 1.0,
               lr_max: float = 0.04,
               tar_cfg_scale: float = 16.5,
               src_cfg_scale: float = 6.0,
               src_prompt_emb: Optional[torch.Tensor] = None, 
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Implementation of Delta Velocity Rectified Flow (DVRF) for InstaFlow
        
        Parameters:
            src_img: Input source image tensor [1, 3, H, W]
            src_prompt: Source prompt string
            tgt_prompt: Target prompt string
            neg_prompt: Negative prompt string
            steps: Number of optimization steps
            eta: Shift coefficient strength
            lr_max: Maximum learning rate
            tar_cfg_scale: Guidance for target
            src_cfg_scale: Guidance for source
            src_prompt_emb: Optional precomputed source prompt embeddings
            tgt_prompt_emb: Optional precomputed target prompt embeddings
        
        Returns:
            img: Edited output image tensor [1, 3, H, W]
        """
        # Prepare Embeddings
        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb) 
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            
            # Combine for efficient batch prediction: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
            combined_embeds = torch.cat([src_neg, src_emb, tgt_neg, tgt_emb], dim=0)

        # Encode Source Image
        with torch.no_grad():
            z_src = self.encode(src_img)

        z_tgt = z_src.clone().detach().requires_grad_(True)

        # Initialize Optimizer
        optimizer = torch.optim.SGD([z_tgt], lr=lr_max)

        # Setup Timesteps
        timesteps = torch.linspace(1.0, 0.0, steps + 1)[:-1].to(self.device)
        
        # Dynamic LR parameters based on linear schedule
        alpha_T_steps = (1.0 / steps) / 1.6 
        alpha_max_dynamic = alpha_T_steps / 1.6 if alpha_T_steps > 0 else lr_max
        beta_tail = alpha_T_steps / 4 if alpha_T_steps > 0 else lr_max / 4

        pbar = tqdm(enumerate(timesteps), total=steps, desc="InstaFlow DVRF Optimization")
        
        for i, t in pbar:
            t_curr = t 
            
            # Update Learning Rate
            current_lr = 2.2 * lr_hump_tail_beta(i+1, steps + 28, alpha_max_dynamic, beta_tail, a=10, b=8)
            optimizer.param_groups[0]["lr"] = current_lr
            
            # Shift coefficient (eta_i)
            eta_i = eta * i / steps
            
            eps = torch.randn_like(z_src)
            
            # Forward Process (Construct noisy latents)
            # Source Branch: qt = (1 - t) * z_src + t * eps
            qt = (1 - t_curr) * z_src + t_curr * eps
            
            # Target Branch (with Shift): pt = (1 - t) * z_tgt + t * eps + eta_i * t * (z_tgt - z_src)
            z_tgt_detached = z_tgt.detach()
            shift_term = eta_i * t_curr * (z_tgt_detached - z_src)
            pt = (1 - t_curr) * z_tgt_detached + t_curr * eps + shift_term

            # Predict Velocities
            with torch.inference_mode():
                # Prepare batch: [qt, qt, pt, pt] corresponding to [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
                latent_input = torch.cat([qt, qt, pt, pt])
                
                # Run U-Net once
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                
                # Split outputs
                v_src_neg, v_src_cond, v_tar_neg, v_tar_cond = v_all.chunk(4)
                
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)
                v_tar = v_tar_neg + tar_cfg_scale * (v_tar_cond - v_tar_neg)

            # Calculate Gradient Approximation
            # grad = (v_tar - v_src) + (1 - eta_i) * (z_tgt - z_src)
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