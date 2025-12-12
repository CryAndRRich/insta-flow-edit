from typing import Dict
import torch

from transformers import T5EncoderModel, BitsAndBytesConfig
from diffusers import StableDiffusion3Pipeline, StableDiffusionPipeline, FluxTransformer2DModel, FluxPipeline


def load_model(model_name: str,
               params_list: Dict[str, Dict[str, str]]) -> StableDiffusion3Pipeline | StableDiffusionPipeline | FluxPipeline:
    """
    Hàm load pipeline dựa trên tên model (SD3, FLUX, INSTAFLOW)
    """
    # Kiểm tra model hợp lệ
    if model_name not in params_list:
        raise ValueError(f"Model '{model_name}' không tồn tại trong params_list. Chọn: {list(params_list.keys())}")

    model_params = params_list[model_name]
    pipe = None

    if model_name == "SD3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_params["model_id"], 
            torch_dtype=torch.float16
        )
        print("- SD3: Kích hoạt Sequential CPU Offload...")
        pipe.enable_sequential_cpu_offload()

    elif model_name == "FLUX":
        print("- FLUX: Đang load với 4-bit Quantization (NF4)...")
        
        # Cấu hình Quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load Transformer (4-bit)
        transformer = FluxTransformer2DModel.from_pretrained(
            model_params["model_id"],
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16
        )

        # Load T5 Encoder (4-bit)
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_params["model_id"],
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
            device_map={"": 0} 
        )

        # Tạo Pipeline
        pipe = FluxPipeline.from_pretrained(
            model_params["model_id"],
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.float16
        )
        
        print("- FLUX: Chuyển Pipeline sang CUDA...")
        pipe.to("cuda")

    elif model_name == "INSTAFLOW":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_params["model_id"], 
            torch_dtype=torch.float16
        )
        pipe.safety_checker = None 
        print("- InstaFlow: Kích hoạt Model CPU Offload...")
        pipe.enable_model_cpu_offload()

    # Kích hoạt VAE Slicing chung cho tất cả model để tiết kiệm VRAM khi decode ảnh lớn
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    elif hasattr(pipe.vae, "enable_slicing"):
        try:
            pipe.vae.enable_slicing()
        except:
            pass

    return pipe