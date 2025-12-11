from typing import Optional, Any
from tqdm.auto import tqdm

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from .utils import scale_noise, calculate_shift


def calc_v_flux(pipe: Any, 
                latents: torch.Tensor, 
                prompt_embeds: torch.Tensor, 
                pooled_prompt_embeds: torch.Tensor, 
                guidance: Optional[torch.Tensor], 
                text_ids: torch.Tensor, 
                latent_image_ids: torch.Tensor, 
                t: torch.Tensor) -> torch.Tensor:
    """
    Tính toán dự đoán vận tốc từ FLUX
    
    Parameters:
        pipe: FLUX Pipeline
        latents: Latent input
        prompt_embeds: Text embeddings chính (T5)
        pooled_prompt_embeds: Text embeddings pooled (CLIP)
        guidance: Tensor chứa giá trị guidance scale (FLUX dùng guidance embed, không dùng CFG chunking truyền thống)
        text_ids, latent_image_ids: Các ID vị trí cho cơ chế RoPE của FLUX
        t: Timestep hiện tại

    Returns:
        torch.Tensor: Vận tốc dự đoán
    """
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        vel_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000, # Chuẩn hóa t về [0, 1]
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
    return vel_pred


@torch.no_grad()
def FlowEditFLUX(pipe: Any,
                 scheduler: Any,
                 x_src: torch.Tensor,
                 src_prompt: str,
                 tar_prompt: str,
                 T_steps: int = 28,
                 n_avg: int = 1,
                 src_guidance_scale: float = 1.5,
                 tar_guidance_scale: float = 5.5,
                 n_min: int = 0,
                 n_max: int = 24) -> torch.Tensor:
    """
    Thuật toán FlowEdit áp dụng cho FLUX
    Logic tương tự SD3 nhưng xử lý thêm phần Packed Latents (nén không gian thành chuỗi)
    """
    device = x_src.device
    
    # Tính kích thước gốc của ảnh
    orig_height = x_src.shape[2] * pipe.vae_scale_factor
    orig_width = x_src.shape[3] * pipe.vae_scale_factor
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # Chuẩn bị Latents: Pack latents thành dạng sequence để đưa vào Transformer
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0], 
        num_channels_latents=num_channels_latents, 
        height=orig_height, 
        width=orig_width, 
        dtype=x_src.dtype, 
        device=device, 
        generator=None, 
        latents=x_src
    )
    # Pack latents thủ công
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3])
    latent_tar_image_ids = latent_src_image_ids # Dùng chung ID ảnh

    # Thiết lập scheduler (FLUX dùng Sigmas thay vì timesteps truyền thống)
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    mu = calculate_shift(
        x_src_packed.shape[1], 
        scheduler.config.base_image_seq_len, 
        scheduler.config.max_image_seq_len, 
        scheduler.config.base_shift, 
        scheduler.config.max_shift
    )
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None, sigmas=sigmas, mu=mu)
    pipe._num_timesteps = len(timesteps)

    # Encode prompts
    (src_prompt_embeds, src_pooled_prompt_embeds, src_text_ids) = pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=device)
    (tar_prompt_embeds, tar_pooled_prompt_embeds, tar_text_ids) = pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, device=device)

    # Xử lý Guidance Embedding
    src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0]) if pipe.transformer.config.guidance_embeds else None
    tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0]) if pipe.transformer.config.guidance_embeds else None

    # Khởi tạo latent chỉnh sửa
    zt_edit = x_src_packed.clone()

    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit FLUX"):
        if T_steps - i > n_max: 
            continue
        
        # Lấy sigma hiện tại và sigma tiếp theo
        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) else t_i
        
        # Giai đoạn 1: Coupled Flow
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src_packed)
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed).to(device)
                
                # Forward process
                zt_src = (1 - t_i) * x_src_packed + (t_i) * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed
                
                # Tính vận tốc cho Source và Target riêng biệt (FLUX không gộp batch như SD3)
                Vt_src = calc_v_flux(pipe, zt_src, src_prompt_embeds, src_pooled_prompt_embeds, src_guidance, src_text_ids, latent_src_image_ids, t)
                Vt_tar = calc_v_flux(pipe, zt_tar, tar_prompt_embeds, tar_pooled_prompt_embeds, tar_guidance, tar_text_ids, latent_tar_image_ids, t)
                
                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)
            
            # Cập nhật Euler
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)
        
        # Giai đoạn 2: Refinement
        else:
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src_packed).to(device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed
            
            Vt_tar = calc_v_flux(pipe, xt_tar, tar_prompt_embeds, tar_pooled_prompt_embeds, tar_guidance, tar_text_ids, latent_tar_image_ids, t)
            xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar
            
            # Nếu là bước cuối, unpack latent ra dạng ảnh 2D để trả về
            if i == len(timesteps) - 1: 
                unpacked_out = pipe._unpack_latents(xt_tar, orig_height, orig_width, pipe.vae_scale_factor)
                return unpacked_out

    # Unpack kết quả cuối cùng
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out