import os
import csv
from typing import Tuple, Optional
from PIL import Image
import torch
import torchvision.transforms as T


def load_data(image_dir: str,
              csv_path: str,
              image_name: str,
              device: str = "cuda",
              dtype: torch.dtype = torch.float16,
              resize_size: int = 1024) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[str]]:
    """
    Loads an image from a directory and retrieves its corresponding prompts from a CSV file

    Parameters:
        image_dir: Path to the folder containing the images.
        csv_path: Path to the CSV file. Expected columns: "name", "source_prompt", "target_prompts".
        image_name: The filename of the image to load (including extension)
        device: Device to put the tensor on ("cuda" or "cpu").
        dtype: Data type of the tensor (float16 or float32).
        resize_size: Size to resize image (default 1024 for SD3/Flux, use 512 for InstaFlow)

    Returns:
        Tuple[Image, str, str]: (image_object, source_prompt, target_prompt)
    """
    image_path = os.path.join(image_dir, image_name)
    image_tensor = None

    if os.path.exists(image_path):
        try:
            pil_image = Image.open(image_path).convert("RGB")

            transforms = T.Compose([
                T.Resize(resize_size, interpolation=T.InterpolationMode.LANCZOS),
                T.CenterCrop(resize_size),
                T.ToTensor(),
                T.Normalize([0.5], [0.5])
            ])

            image_tensor = transforms(pil_image).unsqueeze(0)
            image_tensor = image_tensor.to(device=device, dtype=dtype)

        except Exception as e:
            print(f"Error loading/processing image file: {e}")
            return None, None, None
    else:
        print(f"Error: Image not found at {image_path}")
        return None, None, None

    source_prompt = None
    target_prompt = None

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return image_tensor, None, None

    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            required_columns = {"name", "source_prompt", "target_prompts"}
            if not required_columns.issubset(set(reader.fieldnames)):
                print(f"Error: CSV missing columns. Found: {reader.fieldnames}")
                return image_tensor, None, None

            image_base_name = os.path.splitext(image_name)[0]
            found = False
            for row in reader:
                if row["name"].strip() == image_base_name:
                    source_prompt = row["source_prompt"]
                    target_prompt = row["target_prompts"]
                    found = True
                    break

            if not found:
                print(f"Warning: Image base name '{image_base_name}' not found in CSV")

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return image_tensor, None, None

    return image_tensor, source_prompt, target_prompt