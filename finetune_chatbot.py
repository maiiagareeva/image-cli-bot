import os, json
from typing import List, Optional
from peft import PeftModel
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
from langchain_community.llms import HuggingFacePipeline
import torch
import argparse
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate


from get_topk_evidence import clip_topk_evidence
from few_shots import few_shots_collection
from clip_zero_shot import classify_leaf_condition


# first try
USE_LORA = True 
BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
LORA_MODEL_PATH = "qwen3-1.7b-guanaco"


# second try
# USE_LORA = True 
# BASE_MODEL_ID   = "Qwen/Qwen1.5-4B-Chat"
# LORA_MODEL_PATH = "qwen15-4b-leaf-lora"

# # third try
# USE_LORA = True 
# BASE_MODEL_ID   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# LORA_MODEL_PATH = "meta-llama-3.1-8b-leaf-lora"


# # fourth try
# USE_LORA = True 
# BASE_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
# LORA_MODEL_PATH = "Mistral-7B-Instruct-v0.2-leaf-lora"

# fifth try
# BASE_MODEL_ID = "Qwen/Qwen1.5-7B-Chat"
# LORA_MODEL_PATH  = "qwen1.5-7b-leaf-lora"

print("Starting CLI Chatbot...")
if USE_LORA:
    print(f"Loading fine-tuned model with LoRA adapter: {LORA_MODEL_PATH}")
else:
    print(f"Loading base model: {BASE_MODEL_ID}")
print("Enter 'exit' to quit.")

def extract_text_from_message(msg):
    content = msg.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)

    return str(content)


class Differential(BaseModel):
    candidates: List[str] = Field(..., description="Alternative diagnosis to consider")
    reason_less_likely: str = Field(..., description="Reason this alternative is less likely")

class LeafDiseaseDetection(BaseModel):
    disease: str = Field(..., description="Primary diagnosis")
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="mild | moderate | severe")
    indicators: List[str] = Field(..., min_length=4, description="visual indicators")
    regions: Optional[List[str]] = Field(default=None, description="leaf regions involved")
    differentials: List[Differential] = Field(..., min_length=2, max_length=3)
    recommended_checks: List[str] = Field(..., min_length=3)
    evidence: str = Field(..., description="≥120 words integrated rationale")
    references: Optional[List[str]] = Field(default=None, description="factual bullets")

parser = JsonOutputParser(pydantic_object=LeafDiseaseDetection)

HEALTHY_CLASS_THRESHOLD = float(os.getenv("HEALTHY_CONF_THRESHOLD", "1.01"))
HEALTHY_MARGIN_THRESHOLD = float(os.getenv("HEALTHY_MARGIN_THRESHOLD", "0.2"))


def _clean_evidence_lines(evidence_block: str | None) -> List[str]:
    lines = []
    if not evidence_block:
        return lines
    for line in evidence_block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        lines.append(line)
    return lines


def _build_healthy_guardrail(evidence_lines: List[str],
                             class_probs: dict[str, float]) -> Optional[LeafDiseaseDetection]:
    if not class_probs:
        return None
    healthy_prob = class_probs.get("healthy", 0.0)
    # find top two labels
    sorted_labels = sorted(class_probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = sorted_labels[0]
    runner_up_prob = sorted_labels[1][1] if len(sorted_labels) > 1 else 0.0
    margin = top_prob - runner_up_prob
    if (
        top_label != "healthy"
        or healthy_prob < HEALTHY_CLASS_THRESHOLD
        or margin < HEALTHY_MARGIN_THRESHOLD
    ):
        return None

    disease_keywords = ("spot", "lesion", "mildew", "gall", "blotch", "necrotic", "powdery")
    if any(any(keyword in line.lower() for keyword in disease_keywords) for line in evidence_lines):
        return None

    indicator_text = [
        "uniform green coloration across the leaf lamina",
        "sharp intact veins without chlorosis",
        "smooth surfaces with no fuzz, mildew or galling",
        "serrated margins remain crisp and undamaged"
    ]
    narrative = (
        f"CLIP zero-shot comparison ranks the healthy template highest with probability {healthy_prob:.2f}. "
        "The descriptive captions emphasize even pigmentation, intact lobes, and the absence of lesions, "
        "oil spots, downy growth, or insect swellings. These observations correspond to a leaf that is actively "
        "photosynthesizing without visible stress. Because no prompts mention sporulation or tissue collapse, the "
        "guardrail emits a healthy assessment rather than forcing a disease label. Continue normal scouting routines, "
        "inspect the underside after humid nights, and keep canopy airflow balanced, but no treatment is recommended "
        "based on this photograph alone."
    )

    return LeafDiseaseDetection(
        disease="healthy",
        confidence=min(0.99, max(0.55, healthy_prob)),
        severity="none",
        indicators=indicator_text,
        regions=["entire leaf"],
        differentials=[
            Differential(
                candidates=["downy_mildew_early_leaf_top"],
                reason_less_likely="No oil-like spots, chlorosis, or downy fuzz are visible in the evidence captions."
            ),
            Differential(
                candidates=["phylloxera"],
                reason_less_likely="There are no dome-shaped galls or blistered protrusions on the blade."
            ),
        ],
        recommended_checks=[
            "continue regular scouting over the next 7–10 days",
            "inspect underside surfaces after humid or rainy weather",
            "maintain standard canopy airflow and sanitation practices"
        ],
        evidence=narrative,
        references=[
            "Healthy grape leaves keep uniform pigment and intact serrations",
            "Absence of oil spots, fuzz, or galls indicates no active infection"
        ]
    )
def diagnosis_to_json_str(diagnosis) -> str:
    if isinstance(diagnosis, BaseModel):
        try:
            return diagnosis.model_dump_json(indent=2, ensure_ascii=False)
        except TypeError:
            return diagnosis.json(indent=2, ensure_ascii=False)

    return json.dumps(diagnosis, indent=2, ensure_ascii=False)












def load_llm(base_model_id=BASE_MODEL_ID,lora_model_path=LORA_MODEL_PATH):
    base_model_instance = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    if USE_LORA:
        model = PeftModel.from_pretrained(base_model_instance, lora_model_path)
        model = model.merge_and_unload()
    else:
        model=base_model_instance
        
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

llm=load_llm()



def build_prompt_text(
                   user_input: str,
                   evidence: Optional[List[str]] = None,
                   shots: Optional[list[tuple[HumanMessage, AIMessage]]] = None,
                   class_probs: Optional[dict[str, float]] = None,
):
    ev = "\n".join(f"- {e}" for e in (evidence or []))

    sys_content = (
    "You are an expert grape-leaf pathologist.\n"
    "The disease has already been identified using visual similarity.\n"
    "You MUST NOT change the disease label.\n"
    "You MUST base indicators only on provided visual evidence.\n"
    "If the disease is 'healthy', severity must be 'none'.\n"
    "Only output 'healthy' when the evidence explicitly states there are no lesions, discoloration, or deformities.\n"
    "If the disease is 'unknown' or 'uncertain', confidence must be <= 0.5.\n"
    "Return STRICTLY a single JSON object matching the schema.\n"
    "Do not add extra diseases or speculation.\n"
    "The field 'indicators' must be a list of short visual cues"
    "(e.g. 'yellow oil-like spots', 'white downy growth')."
    "Do NOT copy the CLIP evidence text verbatim."
)


    schema_hint = (
        "JSON fields:\n"
        "  disease (string), confidence (0-1 float), severity (mild|moderate|severe),\n"
        "  indicators (>=4 short bullet strings), regions (optional list),\n"
        "  differentials (2-3 objects: {candidates, reason_less_likely}),\n"
        "  recommended_checks (>=3 items),\n"
        "  evidence (>=120 words, one paragraph, objective visual rationale),\n"
        "  references (optional short factual bullets).\n"
    )


    shots_text = ""
    if shots:
        for i, (h, a) in enumerate(shots, start=1):
            h_text = extract_text_from_message(h)
            a_text = extract_text_from_message(a)

            h_text = h_text[:800]
            a_text = a_text[:1200]

            shots_text += f"### Example {i}\n"
            shots_text += f"User:\n{h_text}\n\n"
            shots_text += f"Assistant (JSON):\n{a_text}\n\n"


    humsn_msg = (
        "### Current Query\n"
        f"{user_input}\n\n"
        f"{schema_hint}"
        "Use only information visible on the leaf plus the bullets below.\n"
        "Top-K visual evidence:\n"
        f"{ev}\n\n"
        "Output: JSON ONLY.\n"
    )

    prompt = sys_content + shots_text + humsn_msg

    return prompt


def identify_run(image_path: str,
                 user_input: str = "Identify the grape leaf disease.",
                 shots_k: int = 4):

    evidence_block = clip_topk_evidence(image_path, k=3)
    evidence_lines = _clean_evidence_lines(evidence_block)
    shots = few_shots_collection(k=shots_k)
    class_probs = classify_leaf_condition(image_path)

    healthy_override = _build_healthy_guardrail(evidence_lines, class_probs)
    if healthy_override:
        return healthy_override

    prompt = build_prompt_text(
        user_input,
        evidence=evidence_lines,
        shots=shots,
        class_probs=class_probs
    )

    model_output = llm.invoke(prompt)
    return parser.invoke(model_output)









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

explain_chain = explain_prompt | llm | StrOutputParser()

def run_explanation(diagnosis, question: str) -> str:
    report_str = diagnosis_to_json_str(diagnosis)

    answer = explain_chain.invoke(
        {
            "report": report_str,
            "question": question,
        }
    )
    return answer










# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--image", required=True, help="Path to grape leaf image")
#     ap.add_argument("--shots_k", type=int, default=4, help="Number of few-shots to use")
#     args = ap.parse_args()

#     image_path = args.image
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image not found: {image_path}")

#     print(f"\nUsing image: {image_path}")
#     print("Enter your question about this leaf. Type 'exit' to quit.\n")


#     while True:
#         try:
#             user_input = input("\nYou: ")

#             if not user_input:
#                 continue

#             if user_input.lower() == 'exit':
#                 print("Exiting chatbot.")
#                 break
            
#             print("Assistant:", end=" ", flush=True)
#             result = langchain_run(image_path,user_input,shots_k=args.shots_k)
            
#             if isinstance(result, BaseModel):
#                 try:
#                     print(result.json(indent=2, ensure_ascii=False))
#                 except TypeError:
#                     print(result.model_dump_json(indent=2, ensure_ascii=False))
#             else:
#                 print(json.dumps(result, indent=2, ensure_ascii=False))

#         except KeyboardInterrupt:
#             print("\nExiting chatbot.")
#             break
#         except Exception as e:
#             print(f"\nAn error occurred: {e}")
#             break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to grape leaf image")
    ap.add_argument("--shots_k", type=int, default=4, help="Number of few-shots to use")
    ap.add_argument("--prompt", type=str, default=None, help="Initial prompt")
    args = ap.parse_args()

    image_path = os.path.expanduser(args.image)
    if not os.path.exists(image_path) and not image_path.startswith("/"):
        alt_path = "/" + image_path.lstrip("/")
        if os.path.exists(alt_path):
            image_path = alt_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\nUsing image: {image_path}")
    print("First question will trigger a full diagnosis (JSON).")
    print("Later questions will get natural-language explanations based on that diagnosis.")
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
                diagnosis = identify_run(image_path, user_input, shots_k=args.shots_k)
                last_diagnosis = diagnosis

                print(diagnosis_to_json_str(diagnosis))

            else:
                print("Assistant (explanation):")
                answer = run_explanation(last_diagnosis, user_input)
                print(answer)

        except KeyboardInterrupt:
            print("\nExiting chatbot.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break
