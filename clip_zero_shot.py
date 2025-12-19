import csv
import os
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from metadata_manifest import store as metadata_manifest_csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32",
                                  use_safetensors=True,
                                  dtype="auto",
                                  ).to(device).eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

REPO_ROOT = Path(__file__).resolve().parent
CLIP_DATA_ROOT = Path(os.getenv("CLIP_DATA_ROOT", REPO_ROOT))
PROTOTYPE_SAMPLE_LIMIT = int(os.getenv("CLIP_PROTOTYPE_SAMPLES", "60"))

CLASS_PROMPTS = OrderedDict({
    "downy_early_leaf_bottom": [
        "macro of a grape leaf underside showing white downy mildew fuzz spiraling along veins and pale sporulation pads",
        "bottom of a grape leaf with cottony mildew growth, angular yellow patches between veins, and damp translucent tissue",
        "grape leaf viewed from below with dense white mycelium clustered under veins and faint greasy lesions surrounding them"
    ],
    "downy_early_leaf_top": [
        "grape leaf upper surface covered with yellow angular oil spots bounded by veins, early downy mildew symptoms",
        "top view of grape leaf showing translucent chlorotic blotches, pale lesions, and greasy halo patterns between veins",
        "grape leaf top exhibiting irregular yellow-green patches, faint sporulation shadows, and no underside fuzz visible yet"
    ],
    "healthy": [
        "healthy grape leaf with uniform medium green color, sharp veins, no oil spots, no mildew, no galls, no necrosis",
        "close view of intact grape leaf blade that is symptom-free, evenly green, and lacks any lesions or surface growth",
        "grape leaf in pristine condition: smooth lamina, hydrated texture, serrated margins intact, zero discoloration"
    ],
    "phylloxera": [
        "grape leaf surface covered in dome-shaped insect galls caused by grape phylloxera, clustered blister bumps",
        "underside of a grape leaf showing cratered blister-like galls from phylloxera feeding and puckered tissue",
        "grape leaf heavily infested with raised wart-like swellings, each gall marking insect feeding chambers"
    ],
})

def _resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    repo_candidate = (REPO_ROOT / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return (CLIP_DATA_ROOT / path).resolve()


def _gather_reference_images(max_per_class: int) -> dict[str, list[str]]:
    csv_path = (REPO_ROOT / metadata_manifest_csv).resolve()
    if not csv_path.exists():
        return {}

    buckets: defaultdict[str, list[str]] = defaultdict(list)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label = row.get("class_name")
            image_rel = row.get("image_path")
            if not label or not image_rel:
                continue
            resolved = _resolve_image_path(image_rel)
            if not resolved.exists():
                continue
            buckets[label].append(str(resolved))

    for label, items in buckets.items():
        random.shuffle(items)
        buckets[label] = items[:max_per_class]

    return buckets


@torch.no_grad()
def embed_texts(texts):
    test = processor.tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    for i, t in enumerate(texts):
        ids = test["input_ids"][i]
        n = len(ids)
        if n > 77:
            first_tokens = processor.tokenizer.convert_ids_to_tokens(ids[:40])
            print(f"[CLIP #of tokens debug] idx={i} tokens={n}  EXCEEDS 77")
            print("[CLIP # of tokens debug] first tokens:", first_tokens)
        else:
            print(f"[CLIP #of tokens debug] idx={i} tokens={n}  OK")

    tensor = processor(text=texts, return_tensors="pt", padding=True,
                                  truncation=True,
                                  max_length=77 ).to(device)
    embed = model.get_text_features(**tensor) #[#of texts,embedding dimension]
    return embed/embed.norm(p=2, dim=-1, keepdim=True)  #[#of texts,embedding dimension]

@torch.no_grad()
def embed_images(paths: list[str]):
    imgs = [Image.open(path).convert("RGB") for path in paths]
    tensor = processor(images=imgs, return_tensors="pt").to(device) #[#of images, channel (3 for RGB), H, W]
    embed = model.get_image_features(**tensor) #[#of images, embedding dimension]
    return embed/embed.norm(p=2, dim=-1, keepdim=True)  #[#of images, embedding dimension]


def _compute_class_prototypes(max_per_class: int) -> dict[str, torch.Tensor]:
    reference = _gather_reference_images(max_per_class)
    prototypes: dict[str, torch.Tensor] = {}
    for label, paths in reference.items():
        if not paths:
            continue
        embeddings = []
        batch = 16
        for i in range(0, len(paths), batch):
            chunk = paths[i:i+batch]
            emb = embed_images(chunk)
            embeddings.append(emb)
        if not embeddings:
            continue
        stacked = torch.cat(embeddings, dim=0)
        centroid = stacked.mean(dim=0)
        centroid = centroid / centroid.norm(p=2)
        prototypes[label] = centroid
    return prototypes


CLASS_PROTOTYPES = _compute_class_prototypes(PROTOTYPE_SAMPLE_LIMIT)

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


# existing embed_images/ image_text_topk unchanged

@torch.no_grad()
def classify_leaf_condition(image_path: str, prompts: OrderedDict[str, list[str]] | None = None) -> dict:
    """Return normalized CLIP probabilities for each grape-leaf class."""
    img_emb = embed_images([image_path])
    if CLASS_PROTOTYPES:
        labels = sorted(CLASS_PROTOTYPES.keys())
        proto = torch.stack([CLASS_PROTOTYPES[label] for label in labels])
        logits = (img_emb @ proto.T).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        return OrderedDict((label, float(prob)) for label, prob in zip(labels, probs))

    prompts = prompts or CLASS_PROMPTS
    if not prompts:
        return {}
    flat_texts: list[str] = []
    label_index: list[str] = []
    for label, variants in prompts.items():
        for text in variants:
            flat_texts.append(text)
            label_index.append(label)

    txt_emb = embed_texts(flat_texts)
    logits = model.logit_scale.exp() * (img_emb @ txt_emb.T)

    per_label: OrderedDict[str, list[float]] = OrderedDict((label, []) for label in prompts.keys())
    for idx, label in enumerate(label_index):
        per_label[label].append(float(logits[0, idx]))

    label_logits = OrderedDict(
        (label, max(scores) if scores else float("-inf"))
        for label, scores in per_label.items()
    )
    logits_tensor = torch.tensor(list(label_logits.values()), dtype=torch.float32)
    probs = torch.softmax(logits_tensor, dim=0)

    return OrderedDict(
        (label, float(prob)) for label, prob in zip(label_logits.keys(), probs)
    )


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description="CLIP zero-shot inspector for grape leaves.")
    parser.add_argument("--image", required=True, help="Path to the grape leaf image")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"Image not found: {args.image}")

    probabilities = classify_leaf_condition(args.image)
    print(json.dumps(probabilities, indent=2))
