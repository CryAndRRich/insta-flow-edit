from typing import Optional, Union, Any
import torch


def scale_noise(scheduler: Any, 
                sample: torch.Tensor, 
                timestep: Union[float, torch.Tensor], 
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Thực hiện forward process trong Flow-Matching: 
    Pha trộn ảnh gốc (X) với nhiễu (N) dựa trên sigma tại timestep t
    Công thức: Z_t = (1 - sigma) * X + sigma * N

    Parameters:
        scheduler: Scheduler của pipeline (thường là FlowMatchEulerDiscreteScheduler)
        sample: Latent của ảnh gốc
        timestep: Bước thời gian hiện tại
        noise: Nhiễu

    Returns:
        torch.Tensor: Sample đã được thêm nhiễu
    """
    # Khởi tạo index cho scheduler nếu cần thiết để truy xuất sigmas
    if hasattr(scheduler, "_init_step_index"):
        scheduler._init_step_index(timestep)
    
    # Lấy giá trị sigma (độ lệch chuẩn nhiễu) tương ứng với bước hiện tại
    if hasattr(scheduler, "sigmas"):
        sigma = scheduler.sigmas[scheduler.step_index]
    else:
        # Giả định sigma tuyến tính theo timestep (dùng cho InstaFlow)
        sigma = timestep 

    # Công thức nội suy tuyến tính
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calculate_shift(image_seq_len: int, 
                    base_seq_len: int = 256, 
                    max_seq_len: int = 4096, 
                    base_shift: float = 0.5, 
                    max_shift: float = 1.16) -> float:
    """
    Tính toán giá trị dịch chuyển thời gian (time shift) cho FLUX
    FLUX thay đổi noise schedule dựa trên độ phân giải ảnh để tối ưu hóa việc tạo chi tiết

    Parameters:
        image_seq_len: Số lượng token của ảnh hiện tại (H * W / patch_size)
        base_seq_len: Độ dài chuỗi cơ sở
        max_seq_len: Độ dài chuỗi tối đa
        base_shift: Shift cơ sở
        max_shift: Shift tối đa

    Returns:
        float: Giá trị "mu" dùng để điều chỉnh lịch trình timestep
    """
    # Tính hệ số góc cho việc nội suy shift
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    # Tính bias
    b = base_shift - m * base_seq_len
    # Tính shift cụ thể cho độ dài ảnh này
    mu = image_seq_len * m + b
    return mu