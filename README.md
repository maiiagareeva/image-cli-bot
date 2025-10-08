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

## Mock mode run

## WSL

```bash
export USE_MOCK=1
python app.py --prompt "Classify the leaf disease and explain why." --image examples/leaf.jpg
```

## real API call

```bash
export USE_MOCK=0
python app.py --prompt "..." --image examples/leaf.jpg
```

## CLIP integration

```bash
python metadata_manifest.py

python - <<'PY'
from get_topk_evidence import clip_topk_evidence
print(clip_topk_evidence("examples/leaf.jpg", k=3))
PY

```

## mock

```bash
export USE_MOCK=1
python app.py --prompt "Classify the leaf disease and explain why." --image examples/leaf.jpg
```

## real api call

```bash
export USE_MOCK=0
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
python app.py --prompt "Classify the leaf disease and explain why." --image examples/leaf.jpg
```
