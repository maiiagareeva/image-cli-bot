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

