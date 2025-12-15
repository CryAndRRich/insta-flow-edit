import os
import yaml

import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import timm

class CLIP_I:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def compute_image_similarity(self, 
                                 img_src: torch.Tensor, 
                                 img_tgt: torch.Tensor) -> float:
        # CLIP-I: Cosine similarity giữa embedding của 2 ảnh
        inputs = self.processor(images=[img_src, img_tgt], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        
        # Normalize
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return (features[0] @ features[1]).item()
    

class DINO:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        # Load DINO v1 (ViT-S/16) từ Facebook Research
        self.model = timm.create_model("vit_small_patch16_224.dino", pretrained=True).to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __call__(self, 
                 img_src: torch.Tensor, 
                 img_tgt: torch.Tensor) -> float:
        # img_src, img_tgt: PIL Images
        t_src = self.transform(img_src).unsqueeze(0).to(self.device)
        t_tgt = self.transform(img_tgt).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat_src = self.model(t_src)
            feat_tgt = self.model(t_tgt)
        
        # Cosine similarity
        return F.cosine_similarity(feat_src, feat_tgt).item()
    
def get_prompt_code(yaml_path: str) -> dict:
    code_map = {}
    if not os.path.exists(yaml_path):
        print(f"Error: Không tìm thấy file {yaml_path}")
        return code_map
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for entry in data:
            if not entry: 
                continue
            tgt_codes = entry.get("target_codes", [])
            tgt_prompts = entry.get("target_prompts", [])
            # Map code -> prompt
            for code, prompt in zip(tgt_codes, tgt_prompts):
                code_map[code] = prompt
    except Exception as e:
        print(f"Error parsing YAML: {e}")
    return code_map