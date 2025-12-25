from typing import Optional, Any

import numpy as np
from tqdm.auto import tqdm
from PIL import Image

import torch
import torchvision.transforms.functional as tf
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
    timestep = t.expand(latents.shape[0]).to(latents.dtype)

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
                 mask_image: Image.Image = None,
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
    dtype = x_src.dtype
    
    mask_packed = None
    if mask_image is not None:
        # Resize mask về kích thước latent (H/8, W/8)
        h_lat, w_lat = x_src.shape[2], x_src.shape[3]
        mask_resized = mask_image.resize((w_lat, h_lat), resample=Image.NEAREST)
        
        mask_tensor = tf.to_tensor(mask_resized).to(device) # [1, H, W]
        mask_tensor = (mask_tensor > 0.5).float()
        mask_tensor = mask_tensor.unsqueeze(0) # [1, 1, H, W]
        
        mask_expanded = mask_tensor.repeat(x_src.shape[0], num_channels_latents, 1, 1) # [B, 16, H, W]
        
        # Pack Mask: Biến đổi [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
        mask_packed = pipe._pack_latents(
            mask_expanded, 
            x_src.shape[0], 
            num_channels_latents, 
            h_lat, w_lat
        )
        mask_packed = mask_packed.to(dtype=dtype)
    
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
        dtype=dtype, 
        device=device, 
        generator=None, 
        latents=x_src
    )
    # Pack latents thủ công
    x_src_packed = pipe._pack_latents(
        x_src, 
        x_src.shape[0], 
        num_channels_latents, 
        x_src.shape[2], 
        x_src.shape[3]
    )
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
    timesteps, T_steps = retrieve_timesteps(
        scheduler, 
        T_steps, 
        device, 
        timesteps=None, 
        sigmas=sigmas, 
        mu=mu
    )
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
            v_delta_avg = torch.zeros_like(x_src_packed)
            for _ in range(n_avg):
                noise = torch.randn_like(x_src).to(device, dtype=dtype)
                
                # Forward process
                zt_src = (1 - t_i) * x_src_packed + (t_i) * noise
                zt_src = zt_src.to(dtype=dtype)

                zt_tar = zt_edit + zt_src - x_src_packed
                
                # Tính vận tốc cho Source và Target riêng biệt (FLUX không gộp batch như SD3)
                vt_src = calc_v_flux(
                    pipe, 
                    zt_src, 
                    src_prompt_embeds, 
                    src_pooled_prompt_embeds, 
                    src_guidance, 
                    src_text_ids, 
                    latent_src_image_ids, 
                    t
                )

                vt_tar = calc_v_flux(
                    pipe, 
                    zt_tar, 
                    tar_prompt_embeds, 
                    tar_pooled_prompt_embeds, 
                    tar_guidance, 
                    tar_text_ids, 
                    latent_tar_image_ids, 
                    t
                )
                
                v_delta_avg += (1 / n_avg) * (vt_tar - vt_src)
            
            if mask_packed is not None:
                v_delta_avg = v_delta_avg * mask_packed

            # Cập nhật Euler
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * v_delta_avg
            zt_edit = zt_edit.to(dtype=dtype)
        
        # Giai đoạn 2: Refinement
        else:
            if i == T_steps - n_min:
                noise = torch.randn_like(x_src_packed).to(device, dtype=dtype)

                xt_src = scale_noise(scheduler, x_src_packed, t, noise=noise)
                xt_src = xt_src.to(dtype=dtype)

                xt_tar = zt_edit + xt_src - x_src_packed
            
            vt_tar = calc_v_flux(
                pipe, 
                xt_tar, 
                tar_prompt_embeds, 
                tar_pooled_prompt_embeds, 
                tar_guidance, 
                tar_text_ids, 
                latent_tar_image_ids, 
                t
            )

            if mask_packed is not None:
                vt_src = calc_v_flux(
                    pipe, 
                    xt_src, 
                    src_prompt_embeds, 
                    src_pooled_prompt_embeds,
                    src_guidance, 
                    src_text_ids, 
                    latent_src_image_ids, 
                    t
                )
                
                vt_effective = mask_packed * vt_tar + (1 - mask_packed) * vt_src
            else:
                vt_effective = vt_tar

            xt_tar = xt_tar + (t_im1 - t_i) * vt_effective
            
    # Unpack kết quả cuối cùng
    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(out, orig_height, orig_width, pipe.vae_scale_factor)
    return unpacked_out