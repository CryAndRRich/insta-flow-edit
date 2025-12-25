from typing import Tuple, Any

from tqdm.auto import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf


def calc_v_instaflow(pipe: Any,
                     latents: torch.Tensor, 
                     prompt_embeds: torch.Tensor, 
                     src_guidance_scale: float, 
                     tar_guidance_scale: float, 
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính toán vận tốc từ InstaFlow (SD1.5 architecture)
    InstaFlow là Rectified Flow nhưng output U-Net vẫn là nhiễu do finetune từ SD1.5

    Parameters:
        pipe: StableDiffusionPipeline
        latents: Input latents (batch size = 4)
        prompt_embeds: Text embeddings
        src_guidance_scale: CFG Scale Source
        tar_guidance_scale: CFG Scale Target
        t: Timestep hiện tại

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Velocity nguồn, Velocity đích)
    """
    # InstaFlow dùng timestep range [0, 1000]
    timestep = t * pipe.scheduler.config.num_train_timesteps
    timestep = timestep.to(latents.device).to(latents.dtype)
    timestep = timestep.expand(latents.shape[0]).to(latents.dtype)

    with torch.no_grad():
        # Dự đoán noise
        noise_pred = pipe.unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        step_index = (t * (pipe.scheduler.config.num_train_timesteps - 1)).long()
            
        # Lấy alpha_t từ scheduler
        alpha_prod_t = pipe.scheduler.alphas_cumprod.to(latents.device)[step_index]
        beta_prod_t = 1 - alpha_prod_t
        
        # Chuyển đổi nhiễu thành vận tốc
        alpha_sqrt = alpha_prod_t.view(-1, 1, 1, 1) ** 0.5
        beta_sqrt = beta_prod_t.view(-1, 1, 1, 1) ** 0.5
        velocity = alpha_sqrt * noise_pred - beta_sqrt * latents

        # Chia chunk 4: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
        vel_src_uncond, vel_src_text, vel_tar_uncond, vel_tar_text = velocity.chunk(4)
        
        # Áp dụng CFG
        vel_pred_src = vel_src_uncond + src_guidance_scale * (vel_src_text - vel_src_uncond)
        vel_pred_tar = vel_tar_uncond + tar_guidance_scale * (vel_tar_text - vel_tar_uncond)

    return vel_pred_src, vel_pred_tar


@torch.no_grad()
def FlowEditInstaFlow(pipe: Any,
                      x_src: torch.Tensor,
                      src_prompt: str,
                      tar_prompt: str,
                      mask_image: Image.Image = None,
                      T_steps: int = 25,
                      n_avg: int = 1,
                      src_guidance_scale: float = 1.5,
                      tar_guidance_scale: float = 16.5,
                      n_min: int = 0,
                      n_max: int = 21) -> torch.Tensor:
    """
    Thuật toán FlowEdit áp dụng cho InstaFlow
    """
    device = x_src.device
    dtype = x_src.dtype

    mask_tensor = None
    if mask_image is not None:
        # Resize mask về kích thước latent (H/8, W/8)
        h_lat, w_lat = x_src.shape[2], x_src.shape[3]
        mask_resized = mask_image.resize((w_lat, h_lat), resample=Image.NEAREST)
        
        mask_tensor = tf.to_tensor(mask_resized).to(device)
        mask_tensor = (mask_tensor > 0.5).float()
        mask_tensor = mask_tensor.unsqueeze(0) # [1, 1, H, W]
        
        mask_tensor = mask_tensor.to(dtype=dtype)

    # Encode Prompts (SD1.5 encode trả về tuple 2 giá trị)
    src_embeds, src_neg_embeds = pipe.encode_prompt(src_prompt, device, 1, True, "")
    tar_embeds, tar_neg_embeds = pipe.encode_prompt(tar_prompt, device, 1, True, "")
    
    # Ghép batch: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
    combined_prompt_embeds = torch.cat([src_neg_embeds, src_embeds, tar_neg_embeds, tar_embeds], dim=0).to(dtype=dtype)

    # Tạo lịch trình thời gian tuyến tính từ 1.0 về 0.0
    timesteps = torch.linspace(1.0, 0.0, T_steps + 1)[:-1].to(device)
    
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit InstaFlow"):
        if T_steps - i > n_max: 
            continue
        
        # Tính bước nhảy thời gian dt
        t_curr = t
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
        dt = abs(t_next - t_curr).to(dtype=dtype) # InstaFlow dùng dt dương

        # Giai đoạn 1: Coupled Flow
        if T_steps - i > n_min:
            v_delta_avg = torch.zeros_like(x_src)
            for _ in range(n_avg):
                noise = torch.randn_like(x_src).to(device, dtype=dtype)
                
                # Forward process
                zt_src = (1 - t_curr) * x_src + t_curr * noise
                zt_src = zt_src.to(dtype=dtype)
                # Coupling
                zt_tar = zt_edit + zt_src - x_src
                
                # Ghép batch 4: [Src_Uncond, Src_Cond, Tar_Uncond, Tar_Cond]
                latent_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                
                vt_src, vt_tar = calc_v_instaflow(
                    pipe, 
                    latent_input, 
                    combined_prompt_embeds, 
                    src_guidance_scale, 
                    tar_guidance_scale, 
                    t_curr
                )
                v_delta_avg += (1 / n_avg) * (vt_tar - vt_src)

            if mask_tensor is not None:
                v_delta_avg = v_delta_avg * mask_tensor

            # Cập nhật Euler
            zt_edit = zt_edit + dt * v_delta_avg
            
        # Giai đoạn 2: Refinement
        else:
            if i == T_steps - n_min:
                noise = torch.randn_like(x_src).to(device, dtype=dtype)

                xt_src = (1 - t_curr) * x_src + t_curr * noise
                xt_src = xt_src.to(dtype=dtype)

                xt_tar = zt_edit + xt_src - x_src
            
            latent_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
            vt_src, vt_tar = calc_v_instaflow(
                pipe, 
                latent_input, 
                combined_prompt_embeds, 
                src_guidance_scale, 
                tar_guidance_scale, 
                t_curr
            )

            if mask_tensor is not None:
                vt_effective = mask_tensor * vt_tar + (1 - mask_tensor) * vt_src
            else:
                vt_effective = vt_tar
            
            xt_tar = xt_tar + dt * vt_effective

    return zt_edit if n_min == 0 else xt_tar