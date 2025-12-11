from typing import Tuple, Any
from tqdm.auto import tqdm

import torch


def calc_v_instaflow(pipe: Any,
                     latents: torch.Tensor, 
                     prompt_embeds: torch.Tensor, 
                     src_guidance_scale: float, 
                     tar_guidance_scale: float, 
                     t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính toán vận tốc từ InstaFlow (SD1.5 architecture)
    InstaFlow là Rectified Flow nhưng output U-Net vẫn là nhiễu

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
    timestep = timestep.expand(latents.shape[0])

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
                      negative_prompt: str,
                      T_steps: int = 20,
                      n_avg: int = 1,
                      src_guidance_scale: float = 1.5,
                      tar_guidance_scale: float = 2.5,
                      n_min: int = 0,
                      n_max: int = 18) -> torch.Tensor:
    """
    Thuật toán FlowEdit áo dụng cho InstaFlow
    """
    device = x_src.device
    
    # Encode Prompts (SD1.5 encode trả về tuple 2 giá trị)
    src_embeds, src_neg_embeds = pipe.encode_prompt(src_prompt, device, 1, True, negative_prompt)
    tar_embeds, tar_neg_embeds = pipe.encode_prompt(tar_prompt, device, 1, True, negative_prompt)
    
    # Ghép batch: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
    combined_prompt_embeds = torch.cat([src_neg_embeds, src_embeds, tar_neg_embeds, tar_embeds], dim=0)

    # Tạo lịch trình thời gian tuyến tính từ 1.0 về 0.0
    timesteps = torch.linspace(1.0, 0.0, T_steps + 1)[:-1].to(device)
    
    zt_edit = x_src.clone()

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit InstaFlow"):
        if T_steps - i > n_max: 
            continue
        
        # Tính bước nhảy thời gian dt
        t_curr = t
        t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
        dt = abs(t_next - t_curr)

        # Giai đoạn 1: Coupled Flow
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src)
            for k in range(n_avg):
                noise = torch.randn_like(x_src).to(device)
                
                # Forward process
                zt_src = (1 - t_curr) * x_src + t_curr * noise
                # Coupling
                zt_tar = zt_edit + zt_src - x_src
                
                # Ghép batch 4: [Src_Uncond, Src_Cond, Tar_Uncond, Tar_Cond]
                latent_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar])
                
                vt_src, vt_tar = calc_v_instaflow(pipe, latent_input, combined_prompt_embeds, src_guidance_scale, tar_guidance_scale, t_curr)
                V_delta_avg += (1 / n_avg) * (vt_tar - vt_src)

            # Cập nhật Euler
            zt_edit = zt_edit + dt * V_delta_avg
            
        # Giai đoạn 2: Refinement
        else:
            if i == T_steps - n_min:
                noise = torch.randn_like(x_src).to(device)
                xt_src = (1 - t_curr) * x_src + t_curr * noise
                xt_tar = zt_edit + xt_src - x_src
            
            latent_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])
            _, vt_tar = calc_v_instaflow(pipe, latent_input, combined_prompt_embeds, src_guidance_scale, tar_guidance_scale, t_curr)
            
            xt_tar = xt_tar + dt * vt_tar
            if i == len(timesteps) - 1: 
                return xt_tar

    return zt_edit if n_min == 0 else xt_tar