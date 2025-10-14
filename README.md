# Image CLI Bot

CLI chatbot for image cropping and prompt handling.  
Prepares images (512x512) and passes them to LangChain API (handled by teammate).

## Features

- Crop & resize images to 512x512
- CLI with prompt and image arguments
- Ready for LangChain integration

## Setup

## WSL/Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

## windows powershell

```Powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install pillow==11.3.0 python-dotenv langchain-core langchain-openai pytest
```

## real API call

```bash
export USE_MOCK=0
python app.py --prompt "..." --image data/test.jpg
```



# preprocess launch
```bash
python preprocess.py \
  --root "leaf_disease_vlm" \
  --manifest-out metadata_manifest.csv \
  --clean-out metadata_clean.csv \
  --resize 512 \
  --output-images-dir ./processed_vlm_512 \
  --result-out preprocess_result.json
```

# common_words launch
```bash
python common_words.py \
  --root "leaf_disease_vlm" \
  --manifest-out metadata_manifest.csv \
  --clean-out metadata_clean.csv \
  --resize 512 \
  --output-images-dir ./processed_vlm_512 \
  --result-out preprocess_result.json
```


## RUN CLIP AND PIPELINE TEST ALL IN ONCE
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -r requirements.txt
```

```bash
python - <<'PY'
from get_topk_evidence import clip_topk_evidence
print(clip_topk_evidence("examples/leaf.jpg", k=3))
PY
```

```bash
export USE_MOCK=1
python app.py --prompt "Classify the leaf disease and explain why." --image examples/leaf.jpg
```

```bash
python - <<'PY'
import os, sys, csv
sys.path.append(os.getcwd())
from clip_zero_shot import image_text_topk

CSV="metadata_clean.csv"; K=3

def map_from_text(t: str) -> str:
    t = t.lower()
    if "phylloxera" in t: return "phylloxera"
    if "healthy"    in t: return "healthy"
    if "downy"      in t:
        if any(k in t for k in ["underside","bottom","lower", "abaxial"]):
            return "downy_early_leaf_bottom"
        if any(k in t for k in ["upper","top","adaxial"]):
            return "downy_early_leaf_top"
        return "downy"
    return "unknown"

tot=ok1=okK=0
with open(CSV, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        img = r["image_path"]
        if r["split"]!="test" or not os.path.exists(img): 
            continue
        texts=[t for t in r["texts"].split("|") if t.strip()]
        texts+=["healthy leaf: uniform green, no spots, no powder"]
        topk = image_text_topk([img], texts, K)[0]

        preds = [map_from_text(t) for t,_ in topk]
        gt    = r["class_name"].lower()

        ok1 += int(preds[0]==gt)

        coarse_gt = "downy" if gt.startswith("downy") else gt
        coarse_preds = [("downy" if p.startswith("downy") else p) for p in preds]
        okK += int(coarse_gt in coarse_preds)
        tot += 1

print(f"Test={tot}  Acc@1={ok1/max(tot,1):.3f}  Acc@{K}={okK/max(tot,1):.3f}")
PY
```

