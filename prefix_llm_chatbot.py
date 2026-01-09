import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
from peft import PeftModel
import os, json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
import argparse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate

device="cuda" if torch.cuda.is_available() else "cpu"
PREFIX_LEN=20

MAPPING="/users/4/shen0574/lora_demo/qwen3-1.7B-ngld-lora/mapping.pt"
CLIP_NAME="openai/clip-vit-base-patch32"

USE_LORA = True 
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_MODEL_PATH = "qwen3-1.7B-ngld-lora"


print("Starting CLI Chatbot...")
if USE_LORA:
    print(f"Loading fine-tuned model with LoRA adapter: {LORA_MODEL_PATH}")
else:
    print(f"Loading base model: {BASE_MODEL_ID}")
print("Enter 'exit' to quit.")

class MappingNet(nn.Module):
    def __init__(self,d_clip,hidden_size,p):
        super().__init__()
        self.p=p
        self.hidden_size=hidden_size
        self.net=nn.Sequential(
            nn.Linear(d_clip,4*hidden_size),
            nn.GELU(),
            nn.Linear(4*hidden_size,p*hidden_size),
        )
    def forward(self,clip_emb):
        B=clip_emb.shape[0]
        x=self.net(clip_emb)
        x=x.view(B,self.p,self.hidden_size)
        return x
    
def load_qwen(BASE_MODEL_ID,LORA_MODEL_PATH,USE_LORA):
    base=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    if USE_LORA:
        MODEL=PeftModel.from_pretrained(base,LORA_MODEL_PATH)
        MODEL=MODEL.merge_and_unload()
    else:
        MODEL=base
    TOKENIZER=AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    TOKENIZER.pad_token=TOKENIZER.eos_token
    TOKENIZER.padding_side="right"
    MODEL.eval()

    return MODEL,TOKENIZER

def load_clip_mapping(qwen_model):
    qwen_device = next(qwen_model.parameters()).device

    clip_model=CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    clip_processor=CLIPProcessor.from_pretrained(CLIP_NAME)

    d_clip=clip_model.config.projection_dim
    h_qwen=qwen_model.config.hidden_size

    mapping_net=MappingNet(d_clip,h_qwen,PREFIX_LEN).to(qwen_device).eval()

    if MAPPING and os.path.exists(MAPPING):
        sd=torch.load(MAPPING,map_location=qwen_device)
        mapping_net.load_state_dict(sd,strict=True)
    
    return clip_model,clip_processor,mapping_net

@torch.no_grad()
def get_clip_embedding(iamge_path,clip_model,clip_processor):
    clip_dev = next(clip_model.parameters()).device

    image=Image.open(iamge_path).convert("RGB")
    iamge_input=clip_processor(images=image,return_tensors="pt").to(clip_dev)
    image_emb=clip_model.get_image_features(**iamge_input)
    image_emb=image_emb/image_emb.norm(dim=-1,keepdim=True)
    return image_emb

@torch.no_grad()
def generate(image_path,prompt,qwen_model,qwen_tokenizer,clip_model,clip_processor,mapping_net):
    qwen_device=next(qwen_model.parameters()).device
    qwen_dtype  = next(qwen_model.parameters()).dtype

    clip_emb=get_clip_embedding(image_path,clip_model,clip_processor).to(qwen_device)
    map_dtype=next(mapping_net.parameters()).dtype
    clip_emb = clip_emb.to(dtype=map_dtype)
    prefix_emb=mapping_net(clip_emb)
    prefix_emb = prefix_emb.to(dtype=qwen_dtype)

    p_encode=qwen_tokenizer(prompt,return_tensors="pt",add_special_tokens=True)
    input_id=p_encode["input_ids"].to(qwen_device)
    attention_id=p_encode["attention_mask"].to(qwen_device)

    text_emb=qwen_model.get_input_embeddings()(input_id)
    text_emb = text_emb.to(dtype=qwen_dtype)
    
    input_emb=torch.cat([prefix_emb,text_emb],dim=1).to(dtype=qwen_dtype)

    prefix_mask=torch.ones((input_id.size(0),PREFIX_LEN),
                           dtype=attention_id.dtype,
                           device=qwen_device)
    attention_mask=torch.cat([prefix_mask,attention_id],dim=1)

    #========================test============================
    print("clip_emb", clip_emb.shape, clip_emb.dtype, clip_emb.device)
    print("prefix_emb", prefix_emb.shape, prefix_emb.dtype, prefix_emb.device)
    print("text_emb", text_emb.shape, text_emb.dtype, text_emb.device)
    print("input_emb", input_emb.shape, input_emb.dtype, input_emb.device)
    print("qwen_dtype", qwen_dtype)


    output=qwen_model.generate(
        inputs_embeds=input_emb,
        attention_mask=attention_mask,
        max_new_tokens=256,
        temperature=0.1,
        pad_token_id=qwen_tokenizer.eos_token_id,
        eos_token_id=qwen_tokenizer.eos_token_id,
        do_sample=True,
    )

    decoded = qwen_tokenizer.decode(output[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].lstrip()
    return decoded


QWEN_MODEL,QWEN_TOKENSIZER=load_qwen(BASE_MODEL_ID,LORA_MODEL_PATH,USE_LORA)
CLIP_MODEL,CLIP_PROCESSOR,MAPPING_NET=load_clip_mapping(QWEN_MODEL)

LLM = RunnableLambda(lambda d:generate(
                    d["image"],
                    d["prompt"],
                    QWEN_MODEL,
                    QWEN_TOKENSIZER,
                    CLIP_MODEL,
                    CLIP_PROCESSOR,
                    MAPPING_NET,
    ))


class LeafDiseaseDetection(BaseModel):
    disease: str = Field(..., description="Primary diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="mild | moderate | severe")
    indicators: List[str] = Field(..., min_length=4, description="visual indicators")
    regions: Optional[List[str]] = Field(default=None, description="leaf regions involved")
    recommended_checks: List[str] = Field(..., min_length=3)
    evidence: str = Field(..., description="≥120 words integrated rationale")
    references: Optional[List[str]] = Field(default=None, description="factual bullets")

JSON_PARSER = JsonOutputParser(pydantic_object=LeafDiseaseDetection)
STRING_PARSER=StrOutputParser()

def diagnosis_to_json_str(diagnosis):
    if isinstance(diagnosis, BaseModel):
        try:
            return diagnosis.model_dump_json(indent=2, ensure_ascii=False)
        except TypeError:
            return diagnosis.json(indent=2, ensure_ascii=False)

    return json.dumps(diagnosis, indent=2, ensure_ascii=False)


def build_prompt_text(user_input):
    sys_content = (
        "You are an expert grape-leaf pathologist.\n"
        "Return STRICTLY a single JSON object conforming to the given schema "
        "(no preface, no markdown, no extra text). "
        "Be precise and concrete. Avoid speculation beyond visible cues."
    )

    schema_hint = (
        "JSON fields:\n"
        "  disease (string), confidence (0-1 float), severity (mild|moderate|severe),\n"
        "  indicators (>=4 short bullet strings), regions (optional list),\n"
        "  recommended_checks (>=3 items),\n"
        "  evidence (>=120 words, one paragraph, objective visual rationale),\n"
        "  references (optional short factual bullets).\n"
    )

    humsn_msg = (
        "### Current Query\n"
        f"{user_input}\n\n"
        f"{schema_hint}"
        "Output: JSON ONLY.\n"
    )

    prompt = sys_content + humsn_msg
    return prompt


def identify_run(image_path,user_input = "Identify the grape leaf disease."):

    prep = RunnableParallel(
        image=RunnableLambda(lambda d: d["image"]),
        user_input=RunnableLambda(lambda d: d["user_input"]),
    )

    to_prompt = RunnableLambda(
        lambda d: {"image":d["image"],
                "prompt":build_prompt_text(d["user_input"])}
    )

    chain = prep | to_prompt | LLM | JSON_PARSER
    return chain.invoke({"image": image_path, "user_input": user_input})


def run_explanation(image_path, diagnosis, question):
    report_str = diagnosis_to_json_str(diagnosis)

    explain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an experienced viticulture and plant disease management expert.\n"
                    "You will receive a JSON diagnosis result of a grape leaf disease.\n"
                    "Based on that diagnosis, answer the user's follow-up question in clear, "
                    "actionable natural language. Do NOT output JSON, only plain text.\n"
                ),
            ),
            (
                "human",
                (
                    "Here is the diagnosis JSON:\n\n"
                    "{report}\n\n"
                    "User question:\n{question}\n\n"
                    "Please answer in detail, focusing on practical advice and interpretation.\n"
                ),
            ),
        ]
    )

    explain_chain = (
    RunnableLambda(lambda d: {
        "image": d["image"],
        "prompt": explain_prompt.format(report=d["report"], question=d["question"])
    })
    | LLM| STRING_PARSER)
    return explain_chain.invoke({"image": image_path, "report": report_str, "question": question})


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to grape leaf image")
    args = ap.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nUsing image: {image_path}")

    print("Type 'exit' to quit.\n")

    last_diagnosis: Optional[LeafDiseaseDetection] = None

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Exiting chatbot.")
                break

            if last_diagnosis is None:
                print("Assistant (diagnosis JSON):")
                diagnosis = identify_run(image_path, user_input)
                last_diagnosis = diagnosis

                print(diagnosis_to_json_str(diagnosis))

            else:
                print("Assistant (explanation):")
                answer = run_explanation(image_path, last_diagnosis, user_input)
                print(answer)

        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break