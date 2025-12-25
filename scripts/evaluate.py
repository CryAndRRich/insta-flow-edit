import os
from typing import Dict, List
from tqdm.auto import tqdm
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from dreamsim import dreamsim

from utils.load_metrics import *


def calculate_metrics(source_dir: str, 
                      target_dir: str, 
                      yaml_path: str) -> Dict[str, List[float]]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CLIP-T (Text)
    clip_t_scorer = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    
    # CLIP-I (Image)
    clip_i_scorer = CLIP_I(device)
    
    # LPIPS
    lpips_scorer = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    
    # DINO
    dino_scorer = DINO(device)
    
    # DreamSim
    dreamsim_scorer = None
    try:
        dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True)
        dreamsim_scorer = dreamsim_model.to(device)
        dreamsim_prep = dreamsim_preprocess
    except ImportError:
        print("Error: DreamSim chưa được cài đặt, bỏ qua metrics này")

    # Transform cho LPIPS
    to_tensor_lpips = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    prompt_map = get_prompt_code(yaml_path)
    if not os.path.exists(target_dir):
        print(f"Không tìm thấy thư mục {target_dir}")
        return
        
    target_files = [f for f in os.listdir(target_dir) if f.endswith((".png", ".jpg"))]

    if not target_files:
        print("Không tìm thấy ảnh trong thư mục Target")
        return

    scores = {
        "CLIP-T": [],
        "CLIP-I": [],
        "LPIPS": [],
        "DINO": [],
        "DreamSim": []
    }

    print(f"Bắt đầu tính toán trên {len(target_files)} ảnh...")
    
    with torch.no_grad():
        for filename in tqdm(target_files, desc="Eval"):
            # filename dạng: "target_1_black_bear.png"
            base_name = os.path.splitext(filename)[0] # "target_1_black_bear"
            if base_name.startswith("target_"):
                target_code = base_name.replace("target_", "", 1) # "1_black_bear"
            else:
                target_code = base_name

            # Tạo tên file source tương ứng: "source_1_black_bear.png"
            src_filename = f"source_{target_code}.png"
            
            tgt_path = os.path.join(target_dir, filename)
            src_path = os.path.join(source_dir, src_filename)
            
            # Kiểm tra tồn tại
            if not os.path.exists(src_path):
                # Fallback: Thử tìm tên gốc nếu không có prefix source_
                src_path_fallback = os.path.join(source_dir, f"{target_code}.png")
                if os.path.exists(src_path_fallback):
                    src_path = src_path_fallback
                else:
                    print(f"Bỏ qua: Không tìm thấy ảnh gốc cho {filename}")
                    continue
            
            if target_code not in prompt_map:
                print(f"Bỏ qua: Không tìm thấy prompt cho code '{target_code}'")
                continue

            # Load Images
            try:
                img_tgt = Image.open(tgt_path).convert("RGB")
                img_src = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"Lỗi load ảnh: {e}")
                continue
                
            target_prompt = prompt_map[target_code]

            # CLIP-T (Text Alignment)
            t_tgt_clip = transforms.ToTensor()(img_tgt).to(device)
            val_clip_t = clip_t_scorer(t_tgt_clip.unsqueeze(0), [target_prompt])
            scores["CLIP-T"].append(val_clip_t.item() / 100)

            # CLIP-I (Image Identity)
            val_clip_i = clip_i_scorer.compute_image_similarity(img_src, img_tgt)
            scores["CLIP-I"].append(val_clip_i)

            # LPIPS (Perceptual Distance)
            t_src_lpips = to_tensor_lpips(img_src).unsqueeze(0).to(device)
            t_tgt_lpips = to_tensor_lpips(img_tgt).unsqueeze(0).to(device)
            val_lpips = lpips_scorer(t_src_lpips, t_tgt_lpips)
            scores["LPIPS"].append(val_lpips.item())

            # DINO (Structure Consistency)
            val_dino = dino_scorer(img_src, img_tgt)
            scores["DINO"].append(val_dino)

            # DreamSim
            if dreamsim_scorer is not None:
                img_src_ds = dreamsim_prep(img_src).to(device)
                img_tgt_ds = dreamsim_prep(img_tgt).to(device)
                val_ds = dreamsim_scorer(img_src_ds, img_tgt_ds)
                scores["DreamSim"].append(val_ds.item())

    return scores

if __name__ == "__main__":
    from configs.config import Config

    yaml_path = Config.get_yaml_path()
    output_src_dirs, output_tar_dirs = Config.get_output_dirs()

    scores = calculate_metrics(output_src_dirs, output_tar_dirs, yaml_path)

    print("=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 50)
    print(f"Số lượng mẫu: {len(scores['CLIP-T'])}")

    print("-" * 16 + "Càng cao càng tốt" + "-" * 17)
    print(f"CLIP-T  : {np.mean(scores['CLIP-T']):.6f}")
    print(f"CLIP-I  : {np.mean(scores['CLIP-I']):.6f}")
    print(f"DINO    : {np.mean(scores['DINO']):.6f}")
    print("-" * 16 + "Càng thấp càng tốt" + "-" * 16)
    print(f"LPIPS   : {np.mean(scores['LPIPS']):.6f}")
    print(f"DreamSim: {np.mean(scores['DreamSim']):.6f}")

    print("=" * 50)