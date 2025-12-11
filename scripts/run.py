import os
import gc
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch

from utils.load_data import *
from utils.load_models import load_model

from flowedit.sd3 import FlowEditSD3
from flowedit.flux import FlowEditFLUX
from flowedit.instaflow import FlowEditInstaFlow

def test_one(model_name: str,
             csv_path: str,
             yaml_path: str,
             configs: Dict[str, Dict[str, str]],
             image_name: str,
             image_dir: str,
             tar_prompt_index: int = 0) -> None:

    # Lấy URL từ CSV
    img_url = image_dir + image_name + ".png"
    if not img_url:
        img_url = f"Không tìm thấy URL cho ảnh '{image_name}'"
    
    # Lấy Prompt từ YAML
    src_prompt, tgt_prompt, neg_prompt = load_prompt_info(yaml_path, image_name, tar_prompt_index)
    if src_prompt is None:
        print(f"Error: Không tìm thấy thông tin prompt cho ảnh '{image_name}' trong file {yaml_path}")
        return

    print(f"- URL ảnh: {img_url}")
    print(f"- Source Prompt: {src_prompt}...")
    print(f"- Target Prompt: {tgt_prompt}...")

    # Bước 1: Dọn dẹp bộ nhớ
    print("Dọn dẹp bộ nhớ...")
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = configs[model_name]
    
    # Bước 2: Load Model Pipeline
    print(f"Khởi tạo model {model_name} ({params['model_id']})...")

    try:
        pipe = load_model(model_name)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return
    
    scheduler = pipe.scheduler
    
    # Bước 3: Setup tham số ngẫu nhiên
    seed = params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Bước 4: Load ảnh từ URL
    init_image = load_image(img_url)
    
    if init_image is None:
        print("Không tải được ảnh")
        return

    max_size = 1024
    w, h = init_image.size
    if w > max_size or h > max_size:
        init_image.thumbnail((max_size, max_size))
        w, h = init_image.size
        print(f"- Resized ảnh thành {w}x{h}")

    divisible = 8 if model_name == "INSTAFLOW" else 16
    w = w - w % divisible
    h = h - h % divisible
    init_image = init_image.resize((w, h), resample=Image.LANCZOS)
    
    # Preprocess
    if hasattr(pipe, "image_processor"):
        input_image = pipe.image_processor.preprocess(init_image).to(device).half()
    else:
        input_image = pipe.feature_extractor(images=init_image, return_tensors="pt").pixel_values.to(device).half()

    # Bước 5: Encode VAE
    print("Encoding Image...")
    with torch.no_grad():
        if model_name == "INSTAFLOW":
            latents_src = pipe.vae.encode(input_image).latent_dist.mode()
            latents_src = latents_src * pipe.vae.config.scaling_factor
        else:
            latents_src = pipe.vae.encode(input_image).latent_dist.mode()
            latents_src = (latents_src - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    
    # Dọn dẹp VRAM
    input_image = None
    torch.cuda.empty_cache()
    
    latents_src = latents_src.to(device)

    # Bước 6: Chạy FlowEdit
    print(f"Running FlowEdit ({model_name})...")
    
    # Gọi hàm xử lý tương ứng
    if model_name == "SD3":
        latents_out = FlowEditSD3(
            pipe, 
            scheduler, 
            latents_src, 
            src_prompt, 
            tgt_prompt, 
            neg_prompt, 
            params["T_steps"], 
            params["n_avg"], 
            params["src_guidance"],
            params["tar_guidance"],
            params["n_min"], 
            params["n_max"]
        )
    elif model_name == "FLUX":
        latents_out = FlowEditFLUX(
            pipe, 
            scheduler, 
            latents_src, 
            src_prompt, 
            tgt_prompt, 
            ["T_steps"], 
            params["n_avg"], 
            params["src_guidance"], 
            params["tar_guidance"], 
            params["n_min"], 
            params["n_max"]
        )
    elif model_name == "INSTAFLOW":
        latents_out = FlowEditInstaFlow(
            pipe, 
            latents_src, 
            src_prompt, 
            tgt_prompt, 
            neg_prompt, 
            params["T_steps"], 
            params["n_avg"], 
            params["src_guidance"], 
            params["tar_guidance"], 
            params["n_min"], 
            params["n_max"]
        )

    # Bước 7: Decode VAE
    print("Decoding Result...")
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        if model_name == "INSTAFLOW":
            latents_out = latents_out / pipe.vae.config.scaling_factor
        else:
            latents_out = (latents_out / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            
        image_out = pipe.vae.decode(latents_out, return_dict=False)[0]
        result_pil = pipe.image_processor.postprocess(image_out, output_type="pil")[0]

    # Bước 8: Display và Save ảnh
    os.makedirs("outputs/test_one/", exist_ok=True)
    save_name = f"outputs/{image_name}_{model_name}_{tar_prompt_index}.png"
    
    result_pil.save(save_name)
    print(f"Saved to {save_name}")
    
    try:
        _, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(init_image)
        axs[0].set_title("Original")
        axs[0].axis("off")
        axs[1].imshow(result_pil)
        axs[1].set_title(f"n_max = {params['n_max']}")
        axs[1].axis("off")
        plt.show()
    except:
        pass


def test_all(model_name: str,
             csv_path: str,
             yaml_path: str,
             configs: Dict[str, Dict[str, str]],
             image_dir: str,
             output_src_dir: str,
             output_tar_dir: str) -> None:
    # Lấy danh sách toàn bộ tên ảnh
    all_image_names = load_dataset_info(csv_path, take_all=True)
    if not all_image_names:
        print("Không tìm thấy ảnh nào trong file CSV")
        return

    # Lấy toàn bộ map prompt/code
    # Cấu trúc: { "bear": { "1_black_bear": {"source": "...", "target": "..."}, ... }, ... }
    all_prompts_map = load_prompt_info(yaml_path, image_key=all_image_names, take_all=True)
    if not all_prompts_map:
        print("Không load được thông tin prompt")
        return

    # Tạo thư mục output
    os.makedirs(output_src_dir, exist_ok=True)
    os.makedirs(output_tar_dir, exist_ok=True)

    print("Dọn dẹp bộ nhớ trước khi chạy...")
    gc.collect()
    torch.cuda.empty_cache()

    try:
        pipe = load_model(model_name)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return
    
    scheduler = pipe.scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = configs[model_name]

    print(f"Bắt đầu chạy trên {len(all_prompts_map)} ảnh gốc...")
    
    # Duyệt qua từng ảnh gốc trong dictionary prompt
    for image_name, codes_data in tqdm(all_prompts_map.items(), desc="Processing Images"):
        # Load và Tiền xử lý ảnh gốc
        img_url = image_dir + image_name + ".png"
        init_image = load_image(img_url)
        if init_image is None:
            print(f"Bỏ qua '{image_name}': Load ảnh thất bại")
            continue

        max_size = 512 if model_name == "FLUX" else 1024
        w, h = init_image.size
        if w > max_size or h > max_size:
            init_image.thumbnail((max_size, max_size))
            w, h = init_image.size
        
        divisible = 8 if model_name == "INSTAFLOW" else 16
        w = w - w % divisible
        h = h - h % divisible
        init_image = init_image.resize((w, h), resample=Image.LANCZOS)

        # Setup Seed cho ảnh
        seed = params["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Encode VAE
        if hasattr(pipe, "image_processor"):
            input_image_tensor = pipe.image_processor.preprocess(init_image).to(device).half()
        else:
            input_image_tensor = pipe.feature_extractor(images=init_image, return_tensors="pt").pixel_values.to(device).half()

        with torch.no_grad():
            if model_name == "INSTAFLOW":
                latents_src_base = pipe.vae.encode(input_image_tensor).latent_dist.mode()
                latents_src_base = latents_src_base * pipe.vae.config.scaling_factor
            else:
                latents_src_base = pipe.vae.encode(input_image_tensor).latent_dist.mode()
                latents_src_base = (latents_src_base - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        
        del input_image_tensor
        torch.cuda.empty_cache()

        # Chạy FlowEdit cho từng Target Code của ảnh này
        # codes_data có dạng: { "code1": {src, tgt}, "code2": {src, tgt} }
        for target_code, prompts in codes_data.items():
            src_prompt = prompts["source"]
            tgt_prompt = prompts["target"]
            # Negative prompt mặc định rỗng
            neg_prompt = "" 

            # Clone latent gốc để không bị thay đổi cho vòng lặp sau
            latents_src = latents_src_base.clone().to(device)

            # Chạy FlowEdit
            if model_name == "SD3":
                latents_out = FlowEditSD3(
                    pipe, 
                    scheduler, 
                    latents_src, 
                    src_prompt, 
                    tgt_prompt, 
                    neg_prompt, 
                    params["T_steps"], 
                    params["n_avg"], 
                    params["src_guidance"], 
                    params["tar_guidance"], 
                    params["n_min"], 
                    params["n_max"]
                )
            elif model_name == "FLUX":
                latents_out = FlowEditFLUX(
                    pipe, 
                    scheduler, 
                    latents_src, 
                    src_prompt, 
                    tgt_prompt, 
                    params["T_steps"], 
                    params["n_avg"], 
                    params["src_guidance"], 
                    params["tar_guidance"], 
                    params["n_min"], 
                    params["n_max"]
                )
            elif model_name == "INSTAFLOW":
                latents_out = FlowEditInstaFlow(
                    pipe, 
                    latents_src, 
                    src_prompt, 
                    tgt_prompt, 
                    neg_prompt, 
                    params["T_steps"], 
                    params["n_avg"], 
                    params["src_guidance"], 
                    params["tar_guidance"], 
                    params["n_min"], 
                    params["n_max"]
                )

            # Decode và Lưu ảnh
            with torch.no_grad():
                if model_name == "INSTAFLOW":
                    latents_out = latents_out / pipe.vae.config.scaling_factor
                else:
                    latents_out = (latents_out / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                
                image_out = pipe.vae.decode(latents_out, return_dict=False)[0]
                result_pil = pipe.image_processor.postprocess(image_out, output_type="pil")[0]

            # Lưu ảnh Source và Target (Tên file = target_code)
            source_save_path = os.path.join(output_src_dir, f"source_{target_code}.png")
            init_image.save(source_save_path)
            target_save_path = os.path.join(output_tar_dir, f"target_{target_code}.png")
            result_pil.save(target_save_path)

            del latents_out, image_out
            torch.cuda.empty_cache()

    print(f"Hoàn thành tất cả! Kiểm tra thư mục {output_src_dir} và {output_tar_dir}")