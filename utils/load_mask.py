from typing import Optional, Tuple
from PIL import Image, ImageFilter, ImageDraw

def create_box_mask_from_seg(seg_mask: Image.Image) -> Image.Image:
    """
    Tạo bbox mask từ segmentation mask
    
    Parameters:
        seg_mask: Segmentation mask nhị phân
        
    Returns:
        Image.Image: Bbox mask
    """
    bbox = seg_mask.getbbox()
    if not bbox: 
        return seg_mask
    
    box_mask = Image.new("L", seg_mask.size, 0)
    draw = ImageDraw.Draw(box_mask)
    
    # Vẽ hình chữ nhật trên mask
    draw.rectangle(bbox, fill=255)
    
    return box_mask


def get_crop_box(mask: Image.Image, 
                 padding: int = 30) -> Optional[Tuple[int, int, int, int]]:
    """
    Lấy bounding box từ mask với padding
    
    Parameters:
        mask: Mask nhị phân
        padding: Số pixel padding quanh bbox
    
    Returns:
        Optional[tuple]: Bbox (l, t, r, b)
    """
    bbox = mask.getbbox()
    if not bbox: 
        return None
    
    # Thêm padding và giới hạn trong kích thước ảnh
    l, t, r, b = bbox
    w, h = mask.size
    l = max(0, l - padding)
    t = max(0, t - padding)
    r = min(w, r + padding)
    b = min(h, b + padding)
    return (l, t, r, b)


def crop_for_editing(image: Image.Image, 
                     mask: Image.Image, 
                     target_size: int = 512) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[tuple]]:
    """
    Crop ảnh và mask dựa trên bounding box của mask, sau đó resize về target_size

    Parameters:
        image: Ảnh gốc
        mask: Mask nhị phân
        target_size: Kích thước mục tiêu sau khi resize

    Returns:
        Tuple[Optional[Image.Image], Optional[Image.Image], Optional[tuple]]: 
            Ảnh đã crop và resize, mask đã crop và resize, bbox gốc
    """
    box = get_crop_box(mask)
    if not box: 
        return None, None, None
    
    crop_img = image.crop(box)
    crop_mask = mask.crop(box)
    
    crop_img_resized = crop_img.resize((target_size, target_size), resample=Image.LANCZOS)
    crop_mask_resized = crop_mask.resize((target_size, target_size), resample=Image.NEAREST)
    
    return crop_img_resized, crop_mask_resized, box


def paste_back(original_image: Image.Image, 
               edited_crop: Image.Image, 
               mask_full: Image.Image, 
               crop_box: Tuple[int, int, int, int],
               blur_radius: int = 5) -> Image.Image:
    """
    Dán phần ảnh đã chỉnh sửa trở lại ảnh gốc với mask mờ
    
    Parameters:
        original_image: Ảnh gốc
        edited_crop: Phần ảnh đã chỉnh sửa (crop và resize)
        mask_full: Mask đầy đủ của ảnh gốc
        crop_box: Bbox gốc để dán lại
        blur_radius: Bán kính làm mờ mask
    
    Returns:
        Image.Image: Ảnh cuối cùng sau khi dán
    """
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]
    # Resize phần đã chỉnh sửa về kích thước gốc của crop
    edited_crop_orig_size = edited_crop.resize((crop_w, crop_h), resample=Image.LANCZOS)
    
    final_image = original_image.copy()
    
    mask_patch = mask_full.crop(crop_box)
    
    # Làm mờ mask để tạo chuyển tiếp mềm mại
    mask_patch_blurred = mask_patch.filter(ImageFilter.GaussianBlur(blur_radius))
    
    final_image.paste(edited_crop_orig_size, crop_box, mask=mask_patch_blurred)
    return final_image