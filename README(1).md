## 1）venv

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip wheel setuptools
```

## 2） CUDA 12.1 --- PyTorch

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

## 3）

```bash
pip install pillow==11.3.0 python-dotenv langchain-core langchain-openai pytest transformers PyYAML safetensors
pip install "huggingface_hub[cli]" --upgrade
```

```bash
cat > requirements.txt << 'REQ'
pillow==11.3.0
python-dotenv
langchain-core
langchain-openai
pytest
transformers
PyYAML
safetensors
torch==2.5.1+cu121
torchvision==0.20.1+cu121
torchaudio==2.5.1+cu121
REQ
# pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

## 4）LM Studio

- open **OpenAI Compatible Server** → **Start Server**
- copy（`http://127.0.0.1:1234/v1`）
- open `http://127.0.0.1:1234/v1/models` in google

## 5） `.env`

```bash
cat > .env << 'ENV'
USE_MOCK=0
LLM_BACKEND=local
OPENAI_API_KEY=sk-local-dummy
LLM_MODEL=qwen2.5-vl-7b-instruct
OPENAI_BASE_URL=http://127.0.0.1:1234/v1
OPENAI_MODEL=gpt-4o-mini
ENV
```

## 6）run preprocess launch & common_words

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

```bash
python common_words.py \
  --root "leaf_disease_vlm" \
  --manifest-out metadata_manifest.csv \
  --clean-out metadata_clean.csv \
  --resize 512 \
  --output-images-dir ./processed_vlm_512 \
  --result-out preprocess_result.json
```

## 7）

```bash
python metadata_manifest.py
```

## 8）run CLIP evidence

```bash
python - <<'PY'
from get_topk_evidence import clip_topk_evidence
print(clip_topk_evidence("examples/leaf.jpeg", k=3))
PY
```

## 9）run local VLM

```bash
python app.py --backend local --model qwen2.5-vl-7b-instruct \
  --prompt "Identify grape leaf disease." --image examples/leaf.jpeg
```

## 10）run local LLM

```bash
python app.py --backend local --model qwen2.5:7b-instruct \
  --prompt "Identify grape leaf disease." --image examples/leaf.jpeg
```

## 12) openai gpt4o-mini

```bash
python app.py --backend local --model gpt-4o-mini \
  --prompt "Identify the leaf disease." --image examples/leaf.jpeg
```
