from PIL import Image
import json

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

def extract_json(text):
    l=text.find("{")
    r = text.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    try:
        return json.loads(text[l:r+1])
    except Exception:
        return None
    
def find_subsequence(hay,needle):
    if not needle or len(needle)>len(hay):
        return []
    hits=[]
    n=len(needle)
    for i in range(len(hay)-n+1):
        if hay[i:i+n]==needle:
            hits.append(i)
    return hits
