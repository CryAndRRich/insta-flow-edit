import os
import gc
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import torch

from configs.config import Config

from utils.load_data import *
from utils.load_mask import *
from utils.load_models import load_model

from flowedit.sd3 import FlowEditSD3
from flowedit.flux import FlowEditFLUX
from flowedit.instaflow import FlowEditInstaFlow


def test_one(model_name: str,
             image_name: str,
             tar_prompt_index: int = 0,
             yaml_path: str = Config.YAML_PATH,
             image_dir: str = Config.IMAGE_DIR,
             mask_dir: str = Config.MASK_DIR,
             masked: str = None,
             params: Dict[str, Dict[str, str]] = Config.MODEL_PARAMS) -> None:
    """
    Chạy FlowEdit trên 1 ảnh cụ thể với model được chọn
    
    Parameters:
        model_name: Tên model ("SD3", "FLUX", "INSTAFLOW")
        image_name: Tên ảnh (không có đuôi)
        tar_prompt_index: Chỉ số Target Prompt trong YAML
        yaml_path: Đường dẫn file YAML chứa prompt
        image_dir: Thư mục chứa ảnh gốc
        mask_dir: Thư mục chứa mask ảnh
        params: Tham số model
        masked: Loại mask sử dụng (None, "segmentation", "bbox")
    """

    assert model_name in params, f"Model '{model_name}' không được hỗ trợ"
    assert masked in [None, "segmentation", "bbox"], "Tham số 'masked' phải là None, 'segmentation' hoặc 'bbox'"
    
    # Lấy đường dẫn ảnh từ IMAGE DIR
    img_path = f"{image_dir}/{image_name}.png"
    if not os.path.exists(img_path):
        print(f"Không tìm thấy ảnh '{image_name}' trong thư mục {image_dir}")
        return
    
    # Lấy Prompt từ YAML
    src_prompt, tgt_prompt = load_prompt_info(yaml_path, image_name, tar_prompt_index)
    if src_prompt is None:
        print(f"Error: Không tìm thấy thông tin prompt cho ảnh '{image_name}' trong file {yaml_path}")
        return

    print(f"- Ảnh: {img_path}")
    print(f"- Source Prompt: {src_prompt}...")
    print(f"- Target Prompt: {tgt_prompt}...")

    # Bước 1: Dọn dẹp bộ nhớ
    print("Dọn dẹp bộ nhớ...")
    gc.collect()
    torch.cuda.empty_cache()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = params[model_name]
    
    # Bước 2: Load Model Pipeline
    print(f"Khởi tạo model {model_name} ({model_params['model_id']})...")

    try:
        pipe = load_model(model_name, params)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return
    
    scheduler = pipe.scheduler
    
    # Bước 3: Setup tham số ngẫu nhiên
    seed = model_params["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Bước 4: Load ảnh từ URL
    init_image = load_image(img_path)
    
    if init_image is None:
        print("Không tải được ảnh")
        return

    max_size = 1024
    w, h = init_image.size
    if w > max_size or h > max_size:
        init_image.thumbnail((max_size, max_size))
        w, h = init_image.size

    divisible = 8 if model_name == "INSTAFLOW" else 16
    w = w - w % divisible
    h = h - h % divisible
    init_image = init_image.resize((w, h), resample=Image.LANCZOS)

    image_to_process = init_image 
    mask_to_process = None         
    crop_info = None             
    use_crop_mode = False  

    if masked is not None:
        mask_path = f"{mask_dir}/{image_name}_mask.png"
        if not os.path.exists(mask_path):
            print(f"Không tìm thấy mask cho ảnh '{image_name}' trong thư mục {mask_dir}")
            return

        seg_mask = Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST)
        
        # Xử lý loại mask
        if masked == "bbox":
            mask_final = create_box_mask_from_seg(seg_mask)
        else:
            mask_final = seg_mask
        
        crop_img, crop_mask, crop_box = crop_for_editing(init_image, mask_final, target_size=512)
        
        if crop_img:
            image_to_process = crop_img
            mask_to_process = crop_mask
            crop_info = (mask_final, crop_box) 
            use_crop_mode = True
        else:
            print("Mask bị lỗi/rỗng")

    # Preprocess
    if hasattr(pipe, "image_processor"):
        input_image = pipe.image_processor.preprocess(image_to_process).to(device).half()
    else:
        input_image = pipe.feature_extractor(images=image_to_process, return_tensors="pt").pixel_values.to(device).half()

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
            pipe=pipe, 
            scheduler=scheduler, 
            x_src=latents_src, 
            src_prompt=src_prompt, 
            tar_prompt=tgt_prompt, 
            mask_image=mask_to_process if masked is not None else None,
            T_steps=model_params["T_steps"], 
            n_avg=model_params["n_avg"], 
            src_guidance_scale=model_params["src_guidance"],
            tar_guidance_scale=model_params["tar_guidance"],
            n_min=model_params["n_min"], 
            n_max=model_params["n_max"]
        )
    elif model_name == "FLUX":
        latents_out = FlowEditFLUX(
            pipe=pipe, 
            scheduler=scheduler, 
            x_src=latents_src, 
            src_prompt=src_prompt, 
            tar_prompt=tgt_prompt,
            mask_image=mask_to_process if masked is not None else None,
            T_steps=model_params["T_steps"], 
            n_avg=model_params["n_avg"], 
            src_guidance_scale=model_params["src_guidance"], 
            tar_guidance_scale=model_params["tar_guidance"], 
            n_min=model_params["n_min"], 
            n_max=model_params["n_max"]
        )
    elif model_name == "INSTAFLOW":
        latents_out = FlowEditInstaFlow(
            pipe=pipe, 
            x_src=latents_src, 
            src_prompt=src_prompt, 
            tar_prompt=tgt_prompt, 
            mask_image=mask_to_process if masked is not None else None,
            T_steps=model_params["T_steps"], 
            n_avg=model_params["n_avg"], 
            src_guidance_scale=model_params["src_guidance"], 
            tar_guidance_scale=model_params["tar_guidance"], 
            n_min=model_params["n_min"], 
            n_max=model_params["n_max"]
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

    final_result = result_pil 
    
    if use_crop_mode and crop_info is not None:
        mask_full, crop_box = crop_info
        final_result = paste_back(init_image, result_pil, mask_full, crop_box)

    # Bước 8: Display và Save ảnh
    os.makedirs("outputs/test_one/", exist_ok=True)
    save_name = f"outputs/{image_name}_{model_name}_{tar_prompt_index}.png"
    
    final_result.save(save_name)
    print(f"Saved to {save_name}")
    
    try:
        cols = 3 if use_crop_mode else 2
        _, axs = plt.subplots(1, cols, figsize=(15, 6))
        
        axs[0].imshow(init_image)
        axs[0].set_title("Original")
        axs[0].axis("off")
        
        if use_crop_mode:
            mask_full_box, _ = crop_info
            axs[1].imshow(mask_full_box, cmap="gray")
            axs[1].set_title("Mask")
            axs[1].axis("off")
            idx = 2
        else:
            idx = 1
        
        axs[idx].imshow(final_result); axs[idx].set_title("Result")
        axs[idx].axis("off")
        plt.show()
    except Exception as e:
        print(f"Lỗi hiển thị ảnh: {e}")


def test_all(model_name: str,
             csv_path: str = Config.CSV_PATH,
             yaml_path: str = Config.YAML_PATH,
             image_dir: str = Config.IMAGE_DIR,
             mask_dir: str = Config.MASK_DIR,
             masked: str = None,
             params: Dict[str, Dict[str, str]] = Config.MODEL_PARAMS,
             output_src_dir: str = Config.OUTPUT_SRC_DIR,
             output_tar_dir: str = Config.OUTPUT_TAR_DIR) -> None:
    """
    Chạy FlowEdit trên toàn bộ ảnh được liệt kê trong file CSV với model được chọn
    """
    
    assert model_name in params, f"Model '{model_name}' không được hỗ trợ"
    assert masked in [None, "segmentation", "bbox"], "Tham số 'masked' phải là None, 'segmentation' hoặc 'bbox'"

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

    # Tạo thư mục output riêng cho từng trường hợp của masked
    suffix = f"_{masked}" if masked else ""
    final_src_dir = f"{output_src_dir}{suffix}"
    final_tar_dir = f"{output_tar_dir}{suffix}"
    
    os.makedirs(final_src_dir, exist_ok=True)
    os.makedirs(final_tar_dir, exist_ok=True)

    print("Dọn dẹp bộ nhớ trước khi chạy...")
    gc.collect()
    torch.cuda.empty_cache()

    print(f"Khởi tạo model {model_name} ({model_params['model_id']})...")

    try:
        pipe = load_model(model_name, params)
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return
    
    scheduler = pipe.scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_params = params[model_name]

    print(f"Bắt đầu chạy trên {len(all_prompts_map)} ảnh gốc...")
    
    # Duyệt qua từng ảnh gốc trong dictionary prompt
    for image_name, codes_data in tqdm(all_prompts_map.items(), desc="Processing Images"):
        # Load và Tiền xử lý ảnh gốc
        img_path = f"{image_dir}/{image_name}.png"
        init_image = load_image(img_path)
        if init_image is None:
            print(f"Bỏ qua '{image_name}': Load ảnh thất bại")
            continue

        max_size = 1024
        w, h = init_image.size
        if w > max_size or h > max_size:
            init_image.thumbnail((max_size, max_size))
            w, h = init_image.size
        
        divisible = 8 if model_name == "INSTAFLOW" else 16
        w = w - w % divisible
        h = h - h % divisible
        init_image = init_image.resize((w, h), resample=Image.LANCZOS)

        # Setup Seed cho ảnh
        seed = model_params["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        image_to_process = init_image
        mask_to_process = None
        crop_info = None
        use_crop_mode = False

        if masked is not None:
            mask_path = f"{mask_dir}/{image_name}_mask.png"

            if os.path.exists(mask_path):
                seg_mask = Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST)
                
                # Xử lý loại mask
                if masked == "bbox":
                    mask_final = create_box_mask_from_seg(seg_mask)
                else:
                    mask_final = seg_mask
                
                crop_img, crop_mask, crop_box = crop_for_editing(init_image, mask_final, target_size=512)
                
                if crop_img:
                    image_to_process = crop_img
                    mask_to_process = crop_mask
                    crop_info = (mask_final, crop_box)
                    use_crop_mode = True
            else: # Nếu mask rỗng thì tự động bỏ qua
                print(f"Bỏ qua {image_name}: Không tìm thấy mask")

        # Encode VAE
        if hasattr(pipe, "image_processor"):
            input_image_tensor = pipe.image_processor.preprocess(image_to_process).to(device).half()
        else:
            input_image_tensor = pipe.feature_extractor(images=image_to_process, return_tensors="pt").pixel_values.to(device).half()

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

            # Clone latent gốc để không bị thay đổi cho vòng lặp sau
            latents_src = latents_src_base.clone().to(device)

            # Chạy FlowEdit
            if model_name == "SD3":
                latents_out = FlowEditSD3(
                    pipe=pipe, 
                    scheduler=scheduler, 
                    x_src=latents_src, 
                    src_prompt=src_prompt, 
                    tar_prompt=tgt_prompt, 
                    mask_image=mask_to_process if masked is not None else None,
                    T_steps=model_params["T_steps"], 
                    n_avg=model_params["n_avg"], 
                    src_guidance_scale=model_params["src_guidance"], 
                    tar_guidance_scale=model_params["tar_guidance"], 
                    n_min=model_params["n_min"], 
                    n_max=model_params["n_max"]
                )
            elif model_name == "FLUX":
                latents_out = FlowEditFLUX(
                    pipe=pipe, 
                    scheduler=scheduler, 
                    x_src=latents_src, 
                    src_prompt=src_prompt, 
                    tar_prompt=tgt_prompt, 
                    mask_image=mask_to_process if masked is not None else None,
                    T_steps=model_params["T_steps"], 
                    n_avg=model_params["n_avg"], 
                    src_guidance_scale=model_params["src_guidance"], 
                    tar_guidance_scale=model_params["tar_guidance"], 
                    n_min=model_params["n_min"], 
                    n_max=model_params["n_max"]
                )
            elif model_name == "INSTAFLOW":
                latents_out = FlowEditInstaFlow(
                    pipe=pipe, 
                    x_src=latents_src, 
                    src_prompt=src_prompt, 
                    tar_prompt=tgt_prompt, 
                    mask_image=mask_to_process if masked is not None else None,
                    T_steps=model_params["T_steps"], 
                    n_avg=model_params["n_avg"], 
                    src_guidance_scale=model_params["src_guidance"], 
                    tar_guidance_scale=model_params["tar_guidance"], 
                    n_min=model_params["n_min"], 
                    n_max=model_params["n_max"]
                )

            # Decode và Lưu ảnh
            with torch.no_grad():
                if model_name == "INSTAFLOW":
                    latents_out = latents_out / pipe.vae.config.scaling_factor
                else:
                    latents_out = (latents_out / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                
                image_out = pipe.vae.decode(latents_out, return_dict=False)[0]
                result_pil = pipe.image_processor.postprocess(image_out, output_type="pil")[0]
            
            final_result = result_pil
            
            # Nếu dùng Crop Mode, phải Paste lại ảnh gốc
            if use_crop_mode and crop_info is not None:
                mask_final, crop_box = crop_info
                final_result = paste_back(init_image, result_pil, mask_final, crop_box)

            # Lưu ảnh Source và Target (Tên file = target_code)
            source_save_path = os.path.join(final_src_dir, f"source_{target_code}.png")
            init_image.save(source_save_path)
            target_save_path = os.path.join(final_tar_dir, f"target_{target_code}.png")
            final_result.save(target_save_path)

            del latents_out, image_out, final_result
            torch.cuda.empty_cache()

    print(f"Hoàn thành tất cả! Kiểm tra thư mục {final_src_dir} và {final_tar_dir}")