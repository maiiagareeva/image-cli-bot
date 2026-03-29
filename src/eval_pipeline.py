import torch
import json
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.BLIP_Qwen.BLIP import BLIP2Model
from src.BLIP_Qwen.cross_model.projector import MLPProjector
from src.BLIP_Qwen.cross_model.query_mix import QueryMixerBlock
from src.BLIP_Qwen.model import QwenWithBLIPPrefix
from src.metrics import compute_metrics_from_text

# ── Config (from your model_meta.json) ────────────────────────────────────
BASE_MODEL = "Qwen/Qwen3-1.7B"
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
ADAPTER_PATH = "src/trained_model_weights"   #dapter_model.safetensors
WEIGHTS_DIR = "src/trained_model_weights"   #projector.pt, query_mixer.pt
QFORMER_DIM = 768
PREFIX_LEN = 32

if torch.backends.mps.is_available():
    device = 'mps' # Mac GPU
    compute_dtype = torch.float32 # mac struggles with float16 => initialize separate
elif torch.cuda.is_available():
    device = 'cuda' # NVIDIA GPU
    compute_dtype = torch.float16
else:
    device = 'cpu' # fallback for older computers
    compute_dtype = torch.float32
print(f"Running on: {device}")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)
base_qwen = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_cfg,
    device_map={"": 0},
    trust_remote_code=True,
)
base_qwen.config.use_cache = False
qwen = PeftModel.from_pretrained(base_qwen, ADAPTER_PATH)
qwen.eval()

d_qwen = qwen.config.hidden_size
# qwen_dtype = qwen.get_input_embeddings().weight.dtype


blip = BLIP2Model(BLIP2_MODEL, device=device, dtype=compute_dtype)
query_mixer = QueryMixerBlock(
    dim=QFORMER_DIM,
    num_heads=8,
    mlp_ratio=2.0,
    dropout=0.1,
).to(device, dtype=compute_dtype)

projector = MLPProjector(
    in_dim=QFORMER_DIM,
    out_dim=d_qwen,
    hidden_dim=2 * d_qwen,
    use_residual=True,
).to(device, dtype=compute_dtype)

projector.load_state_dict(torch.load(f"{WEIGHTS_DIR}/projector.pt", map_location=device))
query_mixer.load_state_dict(torch.load(f"{WEIGHTS_DIR}/query_mixer.pt", map_location=device))
print("Loaded projector.pt and query_mixer.pt")

# assembling all
model = QwenWithBLIPPrefix(
    qwen=qwen,
    blip=blip,
    projector=projector,
    query_mixer=query_mixer,
    use_weighted_loss=False,
)
model.eval()


processor = AutoProcessor.from_pretrained(BLIP2_MODEL)

def ensure_pil_rgb(image): # same from train
    if isinstance(image,Image.Image):
        return image.convert("RGB")
    if isinstance(image,dict) and "bytes" in image and image["bytes"] is not None:
        from io import BytesIO
        return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if isinstance(image,dict) and "path" in image and image["path"] is not None:
        return Image.open(image["path"]).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)}")

def generate_prediction(image, question: str) -> str:
    image = ensure_pil_rgb(image)

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device, dtype=compute_dtype)

    # tokenize the question
    tok = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    input_ids      = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
	    repetition_penalty=1.3,
	    no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# test set and collect predictions
# with open("test_dataset.json") as f:
#     test_data = json.load(f)
# -----------------------------------------------------------
from datasets import load_dataset

ds = load_dataset("qingwuuu/ngld-grape-leaf-vlm-w-img-without-diff-ref")
test_data = ds["test"]

predictions = []
references  = []
print(f"SAMPLE SIZE = {len(test_data)}")


@torch.no_grad()
def generate_prediction_text_only(question: str) -> str:
    """Bypass vision — test if Qwen+LoRA alone generates correctly"""
    tok = tokenizer(question, return_tensors="pt", truncation=True, max_length=256)
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Call qwen directly, bypassing BLIP+projector
    output_ids = model.qwen.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    input_len = input_ids.shape[1]
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)

# Test it
print("TESSSSSTTTTTTTT#######################")
#print(generate_prediction_text_only(test_data[0]["prompt"]))
print("22222TESSSSSTTTTTTTT#######################")

i = 0
for sample in test_data:
    pred = generate_prediction(sample["image"], sample["prompt"])
    try:
        answer_json = json.loads(sample["answer"])
        reference = answer_json.get("evidence", sample["answer"])
    except (json.JSONDecodeError, TypeError):
        reference = sample["answer"]
    predictions.append(pred)
    references.append(reference)
    print(f"PRED: {pred}")
    print(f"REF : {reference}\n")
    if i == 4:
        break

# metrics.py
results = compute_metrics_from_text(predictions, references)
print("\n Evaluation results:\n")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
