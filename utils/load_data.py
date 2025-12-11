from typing import List, Union, Dict, Tuple
import os
import csv
import yaml
from io import BytesIO
import requests
from PIL import Image


def load_dataset_info(csv_path: str, 
                      image_key: str | None = None, 
                      take_all: bool = False) -> str | List[str] | None:
    """
    Đọc file CSV
    - Nếu take_all=False: Trả về URL của ảnh dựa trên image_key
    - Nếu take_all=True: Trả về danh sách toàn bộ tên ảnh (cột "name") trong CSV
    """
    if not os.path.exists(csv_path):
        print(f"Error: Không tìm thấy file {csv_path}")
        return None
    
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Chuẩn hóa tên cột
            fieldnames = [name.strip() for name in reader.fieldnames]
            
            # Đọc lại file với fieldnames đã chuẩn hóa
            f.seek(0)
            reader = csv.DictReader(f, fieldnames=fieldnames)
            next(reader) # Bỏ qua header
            
            if take_all: # Lấy tất cả ảnh
                all_names = []
                for row in reader:
                    name = row.get("name", "").strip()
                    if name:
                        all_names.append(name)
                print(f"Thu được danh sách tên của {len(all_names)} ảnh")
                return all_names

            # Lấy 1 ảnh
            if image_key is None:
                print("Error: image_key không được để trống khi take_all=False")
                return None

            for row in reader:
                if row["name"].strip() == image_key:
                    return row["url"].strip()

    except Exception as e:
        print(f"Error parsing CSV: {e}")
    
    return None

def load_prompt_info(yaml_path: str, 
                     image_key: Union[str, List[str]], 
                     tar_prompt_index: int = 0,
                     take_all: bool = False) -> Union[Tuple[str | None, str | None, str | None], Dict[str, Dict[str, Dict[str, str]]] | None]:
    """
    Đọc file YAML
    - Nếu take_all=False: Trả về (src_prompt, tgt_prompt, neg_prompt) của 1 ảnh dựa trên image_key và tar_prompt_index
    - Nếu take_all=True: image_key là list tên ảnh. Trả về Dictionary cấu trúc:
      {
          "image_name": {
              "target_code_1": {"source": src_prompt, "target": tgt_prompt_1},
              "target_code_2": {"source": src_prompt, "target": tgt_prompt_2},
              ...
          },
          ...
      }
    """
    if not os.path.exists(yaml_path):
        print(f"Error: Không tìm thấy file {yaml_path}")
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        if take_all: # Lấy tất cả thông tin
            if isinstance(image_key, str):
                target_keys = {image_key}
            else:
                target_keys = set(image_key) # Chuyển list thành set để tìm kiếm nhanh hơn
            
            result_dict = {}

            for entry in data:
                if not entry: 
                    continue
                
                init_img_path = entry.get("init_img", "")
                base_name = os.path.splitext(os.path.basename(init_img_path))[0]
                
                # Chỉ xử lý nếu ảnh nằm trong danh sách yêu cầu
                if base_name in target_keys:
                    src_prompt = entry.get("source_prompt", "")
                    tgt_prompts = entry.get("target_prompts", [])
                    tgt_codes = entry.get("target_codes", [])

                    # Kiểm tra dữ liệu khớp nhau
                    if len(tgt_prompts) != len(tgt_codes):
                        print(f"Warning: Số lượng target_prompts và target_codes không khớp cho ảnh '{base_name}'")
                        tgt_prompts = tgt_prompts[:len(tgt_codes)]

                    # Tạo dictionary mapping: Code -> {Source, Target}
                    img_map = {}
                    for code, tgt_p in zip(tgt_codes, tgt_prompts):
                        img_map[code] = {
                            "source": src_prompt,
                            "target": tgt_p
                        }
                    
                    result_dict[base_name] = img_map

            print(f"Thu được source và target prompt cho {len(result_dict)} ảnh")
            return result_dict

        else: # Lấy 1 source prompt và 1 target prompt
            if not isinstance(image_key, str):
                print("Error: image_key phải là string khi take_all=False")
                return None, None, None

            for entry in data:
                if not entry: 
                    continue
                
                init_img_path = entry.get("init_img", "")
                base_name = os.path.splitext(os.path.basename(init_img_path))[0]
                
                if base_name == image_key:
                    src_prompt = entry.get("source_prompt", "")
                    tgt_prompts = entry.get("target_prompts", [])
                    
                    try:
                        tgt_prompt = tgt_prompts[tar_prompt_index] if tgt_prompts else ""
                    except IndexError:
                        print(f"Error: Index {tar_prompt_index} vượt quá giới hạn cho ảnh '{image_key}'")
                        return None, None, None
                    
                    neg_prompt = entry.get("negative_prompt", "")
                    return src_prompt, tgt_prompt, neg_prompt

    except Exception as e:
        print(f"Error parsing YAML: {e}")

    return None if take_all else (None, None, None)

def load_image(path_or_url: str) -> Image.Image | None:
    """
    Hàm load ảnh từ đường dẫn hoặc URL
    """
    try:
        if os.path.exists(path_or_url):
            img = Image.open(path_or_url).convert("RGB")
        else:
            print(f"Downloading image from {path_or_url}...")
            # Giả lập User-Agent để tránh bị chặn bởi một số server ảnh
            headers = {"User-Agent": "Mozilla/5.0"} 
            response = requests.get(path_or_url, headers=headers, stream=True)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None