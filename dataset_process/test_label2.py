
import os
import csv
import json
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_ROOT = Path("GreenHouse_img")
LABEL_DIR = DATA_ROOT / "labels"
IMAGE_DIRS = [DATA_ROOT / "next118"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def iter_all_images():
    for d in IMAGE_DIRS:
        print(d)
        if not d.exists():
            print('continue')
            continue
        split_name = d.name
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, split_name, p.name, p.stem

rows = []
images = list(iter_all_images())
print(f"Found {len(images)} images across: {[str(d) for d in IMAGE_DIRS]}")

missing_label = 0
for img_path, split_name, filename, stem in tqdm(images):
    label_path = LABEL_DIR / f"{stem}.txt"
    raw = label_path.read_text(encoding="utf-8").strip()
    print(f"raw: {raw}, label_path: {label_path}")
    if raw == 2:
        break