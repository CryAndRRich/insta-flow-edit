from typing import Tuple, Any
import gc
from tqdm.auto import tqdm

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from .utils import scale_noise

def calc_v_sd3(pipe: Any, 
               src_tar_latent_model_input: torch.Tensor, 
               src_tar_prompt_embeds: torch.Tensor, 
               src_tar_pooled_prompt_embeds: torch.Tensor, 
               src_guidance_scale: float, 
               tar_guidance_scale: float, 
               t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tính toán dự đoán vận tốc (velocity prediction) từ SD3

    Parameters:
        pipe: SD3 Pipeline
        src_tar_latent_model_input: Input latent gộp (batch size = 4)
        src_tar_prompt_embeds: Text embeddings gộp
        src_tar_pooled_prompt_embeds: Pooled text embeddings gộp
        src_guidance_scale: Hệ số CFG cho ảnh nguồn
        tar_guidance_scale: Hệ số CFG cho ảnh đích
        t: Timestep hiện tại

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (Velocity nguồn, Velocity đích)
    """
    # Mở rộng timestep t để khớp với batch size của input (thường là 4)
    timestep = t.expand(src_tar_latent_model_input.shape[0])

    with torch.no_grad():
        # Chạy forward pass qua Transformer của SD3
        vel_pred_src_tar = pipe.transformer(
            hidden_states=src_tar_latent_model_input,
            timestep=timestep,
            encoder_hidden_states=src_tar_prompt_embeds,
            pooled_projections=src_tar_pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Chia output thành 4 phần (FlowEdit chạy batch 4: Src_Uncond, Src_Text, Tar_Uncond, Tar_Text)
        src_vel_pred_uncond, src_vel_pred_text, tar_vel_pred_uncond, tar_vel_pred_text = vel_pred_src_tar.chunk(4)
        
        # Áp dụng công thức CFG
        # v_pred = v_uncond + scale * (v_text - v_uncond)
        noise_vel_src = src_vel_pred_uncond + src_guidance_scale * (src_vel_pred_text - src_vel_pred_uncond)
        noise_vel_tar = tar_vel_pred_uncond + tar_guidance_scale * (tar_vel_pred_text - tar_vel_pred_uncond)

    return noise_vel_src, noise_vel_tar


@torch.no_grad()
def FlowEditSD3(pipe: Any,
                scheduler: Any,
                x_src: torch.Tensor,
                src_prompt: str,
                tar_prompt: str,
                negative_prompt: str,
                T_steps: int = 50,
                n_avg: int = 1,
                src_guidance_scale: float = 3.5,
                tar_guidance_scale: float = 13.5,
                n_min: int = 0,
                n_max: int = 15) -> torch.Tensor:
    """
    Thuật toán FlowEdit áp dụng cho SD3
    
    Parameters:
        pipe: SD3 Pipeline
        scheduler: Scheduler của pipeline
        x_src: Latent của ảnh gốc (Source Image)
        src_prompt: Prompt mô tả ảnh gốc
        tar_prompt: Prompt mô tả ảnh đích mong muốn
        negative_prompt: Prompt mô tả ảnh đích không mong muốn
        T_steps: Tổng số bước lấy mẫu
        n_avg: Số lần lấy trung bình nhiễu tại mỗi bước (để ổn định ODE)
        src_guidance_scale: CFG scale cho source
        tar_guidance_scale: CFG scale cho target
        n_min: Bước dừng thuật toán coupled (chuyển sang sinh ảnh thường)
        n_max: Bước bắt đầu thực hiện chỉnh sửa (bỏ qua các bước nhiễu đầu tiên)

    Returns:
        torch.Tensor: Latent của ảnh kết quả (Target Image)
    """
    device = x_src.device
    
    # Lấy danh sách timesteps từ scheduler (từ 1000 về 0)
    timesteps, T_steps = retrieve_timesteps(scheduler, T_steps, device, timesteps=None)
    pipe._num_timesteps = len(timesteps)
    
    # Dọn dẹp bộ nhớ trước khi encode (tránh OOM)
    torch.cuda.empty_cache()
    gc.collect()

    # Encode prompt nguồn (Source)
    # Ở đây do_classifier_free_guidance=True để lấy cả negative embeds cho CFG
    (src_prompt_embeds, src_negative_prompt_embeds, src_pooled_prompt_embeds, src_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=src_prompt, 
        prompt_2=None, 
        prompt_3=None, 
        negative_prompt=negative_prompt, 
        do_classifier_free_guidance=True, 
        device=device
    )
    
    torch.cuda.empty_cache()

    # Encode prompt đích (Target)
    (tar_prompt_embeds, tar_negative_prompt_embeds, tar_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=tar_prompt, 
        prompt_2=None, 
        prompt_3=None, 
        negative_prompt=negative_prompt, 
        do_classifier_free_guidance=True, 
        device=device
    )
    
    torch.cuda.empty_cache() 
    
    # Ghép các embeddings lại để xử lý batch: [Neg_Src, Pos_Src, Neg_Tar, Pos_Tar]
    src_tar_prompt_embeds = torch.cat([src_negative_prompt_embeds, src_prompt_embeds, tar_negative_prompt_embeds, tar_prompt_embeds], dim=0)
    src_tar_pooled_prompt_embeds = torch.cat([src_negative_pooled_prompt_embeds, src_pooled_prompt_embeds, tar_negative_pooled_prompt_embeds, tar_pooled_prompt_embeds], dim=0)
    
    # Khởi tạo latent cần chỉnh sửa Z_edit (ban đầu là ảnh gốc X_src)
    zt_edit = x_src.clone()

    # Vòng lặp lấy mẫu ngược (từ t=1 về t=0)
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="FlowEdit SD3"):
        # Bỏ qua các bước đầu nếu chưa đến n_max (giữ nguyên ảnh gốc)
        if T_steps - i > n_max:
            continue
        
        # Chuẩn hóa t về khoảng [0, 1] cho tính toán ODE
        t_i = t / 1000
        # Lấy t của bước tiếp theo
        t_im1 = (timesteps[i + 1]) / 1000 if i + 1 < len(timesteps) else torch.zeros_like(t_i).to(t_i.device)
        
        # Giai đoạn 1: Coupled Flow
        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src)
            
            # Lặp n_avg lần để lấy trung bình nhiễu (giảm phương sai)
            for k in range(n_avg):
                # Tạo nhiễu Gaussian ngẫu nhiên N(0,1)
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                
                # Tạo phiên bản nhiễu của ảnh gốc tại thời điểm t (Forward Process)
                # Công thức: Z_t^src = (1 - t) * X + t * N
                zt_src = (1 - t_i) * x_src + (t_i) * fwd_noise

                # Tạo phiên bản nhiễu giả định của ảnh đích dựa trên zt_edit hiện tại (Coupling)
                zt_tar = zt_edit + zt_src - x_src

                # Chuẩn bị input cho model: ghép [Src_Uncond, Src_Cond, Tar_Uncond, Tar_Cond]
                src_tar_latent_model_input = torch.cat([zt_src, zt_src, zt_tar, zt_tar]) 

                # Tính vận tốc cho Source và Target từ model
                Vt_src, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

                # Tính Delta Velocity: Hướng thay đổi từ Src -> Tar
                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            # Cập nhật Euler: Di chuyển Z_edit theo hướng Delta V
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        # Giai đoạn 2: Refinement
        else:
            # Nếu vừa chuyển giao giai đoạn, cần tái tạo nhiễu chuẩn
            if i == T_steps - n_min:
                fwd_noise = torch.randn_like(x_src).to(x_src.device)
                xt_src = scale_noise(scheduler, x_src, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src
                
            # Chỉ tập trung vào sinh ảnh Target
            src_tar_latent_model_input = torch.cat([xt_tar, xt_tar, xt_tar, xt_tar])

            # Tính vận tốc Target
            _, Vt_tar = calc_v_sd3(pipe, src_tar_latent_model_input, src_tar_prompt_embeds, src_tar_pooled_prompt_embeds, src_guidance_scale, tar_guidance_scale, t)

            # Cập nhật Euler cho Target
            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample
        
    return zt_edit if n_min == 0 else xt_tar