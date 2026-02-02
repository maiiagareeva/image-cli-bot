from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)

from transformers import Blip2Processor
import json
from PIL import Image
import torch
import yaml

config=yaml.safe_load(open("configs/blip2.yaml","r",encoding="utf-8"))
model_id=config.get("model_id", "Salesforce/blip2-opt-2.7b")
image_path=config["image_path"]
use_fewshot=config.get("use_fewshot")
gen=config.get("generation",{})
max_new_tokens=gen.get("max_new_tokens")
do_sample=gen.get("do_sample")
num_beams=gen.get("num_beams")

def get_image(path):
    image=Image.open(path).convert("RGB")
    return image

SCHEMA = r"""
Output a diagnosis in the following JSON schema:
{
  "disease": "<one of: Healthy | Downy Mildew | Unknown>",
  "indicators": ["<short visual cues>", "..."],
  "recommended_checks": ["<short follow-up checks>", "..."],
  "evidence": "<a concise but complete justification grounded in visible symptoms>"
}
""".strip()

FEWSHOT = r"""
Example 1 (Healthy):
{
  "disease": "Healthy",
  "indicators": [
    "uniform vibrant green coloration",
    "no visible lesions, spots, or oil-like patches",
    "intact venation without distortion",
    "no chlorosis or necrotic areas"
  ],
  "recommended_checks": [
    "monitor periodically for early symptom emergence",
    "inspect leaf underside for subtle mycelial growth",
    "check neighboring leaves for early-stage infection signs"
  ],
  "evidence": "The leaf appears uniformly green with no visible lesions, angular yellowing, oil-like spots, or necrosis. Veins look intact and there is no surface pattern consistent with downy mildew. Based on the absence of disease hallmarks in the visible regions, the most consistent diagnosis is healthy."
}

Example 2 (Downy Mildew):
{
  "disease": "Downy Mildew",
  "indicators": [
    "irregular yellow-green (chlorotic) patches often bounded by veins",
    "possible oil-like lesions on the upper surface",
    "potential whitish/grayish growth on the underside (may not be visible depending on view)",
    "localized tissue discoloration consistent with early infection"
  ],
  "recommended_checks": [
    "inspect the underside under good lighting for white/gray sporulation",
    "compare multiple leaves to confirm consistent lesion patterns",
    "monitor progression over 24–72 hours for expansion and necrosis"
  ],
  "evidence": "The observed discoloration pattern is consistent with early downy mildew presentation (chlorotic/oil-like patches that may align with venation). While underside sporulation is not always captured in a single photo, the visible symptom pattern supports downy mildew over a healthy leaf."
}
""".strip()
def build_pompt(use_fewshot=False):
    prompt=(
        "You are a grape leaf disease diagnosis assistant. "
        "Analyze the image and output a diagnosis in the following JSON schema.\n\n"
        + SCHEMA
        + "\n"
    )
    if use_fewshot:
        prompt+="\n"+FEWSHOT+"\n"
    prompt += "\n\nNow analyze the image and output ONLY the JSON object."
    return prompt


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id, 
    load_in_8bit=True, 
    device_map="auto", 
    # dtype=torch.float16
)
# model.to(device)
model.eval()

image=get_image(image_path)

prompt = build_pompt(False)
inputs = processor(images=image, 
                   text=prompt, 
                   return_tensors="pt"
                   )
if torch.cuda.is_available():
    inputs={k:v.to("cuda") for k,v in inputs.items()}

generated_ids = model.generate(**inputs,
                               max_new_tokens=max_new_tokens,
                               do_sample=do_sample,
                               num_beams=(num_beams if not do_sample else 1)
                               )
# generated_texts = processor.tokenizer.batch_decode(
#     generated_ids,
#     skip_special_tokens=True
# )
# print(generated_texts[0].strip())
input_len = inputs["input_ids"].shape[1]

text = processor.tokenizer.decode(
      generated_ids[0][input_len:], 
      skip_special_tokens=True
    ).strip()
print(text)
