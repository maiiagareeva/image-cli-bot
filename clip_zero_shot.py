import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@torch.no_grad()
def embed_texts(texts):
    tensor = processor(text=texts, return_tensors="pt", padding=True).to(device) #[#of texts, max_length]
    embed = model.get_text_features(**tensor) #[#of texts,embedding dimension]
    return embed/embed.norm(p=2, dim=-1, keepdim=True)  #[#of texts,embedding dimension]

@torch.no_grad()
def embed_images(paths: list[str]):
    imgs = [Image.open(path).convert("RGB") for path in paths]
    tensor = processor(images=imgs, return_tensors="pt").to(device) #[#of images, channel (3 for RGB), H, W]
    embed = model.get_image_features(**tensor) #[#of images, embedding dimension]
    return embed/embed.norm(p=2, dim=-1, keepdim=True)  #[#of images, embedding dimension]

@torch.no_grad()
def image_text_topk(image_paths: list[str], texts: list[str], k: int = 3):
    if len(texts) == 0 or len(image_paths) == 0:
        return []
    txt = embed_texts(texts)
    imgs = embed_images(image_paths)

    logits = model.logit_scale.exp() * (imgs @ txt.T) 
    k = min(k, logits.size(1))
    vals, idx = torch.topk(logits, k=k, dim=1)
    results= []
    for n in range(logits.size(0)):
        row = [(texts[int(i)], float(vals[n, j])) for j, i in enumerate(idx[n])]
        results.append(row)
    return results
