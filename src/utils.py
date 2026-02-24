from PIL import Image

def ensure_pil_rgb(image):
    if isinstance(image,Image.Image):
        return image.convert("RGB")
    if isinstance(image,dict) and "bytes" in image and image["bytes"] is not None:
        from io import BytesIO
        return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image,dict) and "path" in image and image["path"] is not None:
        return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)}")

def set_requires_grad(module,flag):
    for p in module.parameters():
        p.requires_grad=flag