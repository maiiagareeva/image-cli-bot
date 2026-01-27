from PIL import Image #open and process image
from PIL import UnidentifiedImageError #catch errors
from pathlib import Path

def center_crop_to_512(src_path: str, dst_path: str | None = None) -> str:
    p = Path(src_path)
    if dst_path is None:
        #why using PNG: (just for now, we would make change according to real situation)
        #lossless
        dst_path = str(p.with_name(p.stem + "_512.png"))
    try:
        #for normalization
        img = Image.open(p).convert("RGB")
    except UnidentifiedImageError:
        raise SystemExit(
            f"[image_utils] Cannot identify image file: {src_path}\n"
            f"makesure this is a PNG/JPEG/JPG format image, but not  HEIC/WEBP/HTML/empty file\n"
        )
    
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
    img = img.resize((512, 512))
    #optimize: optimization to reduce file size
    img.save(dst_path, "PNG", optimize=True)
    return dst_path
