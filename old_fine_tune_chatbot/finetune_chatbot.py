import os, json
from typing import List, Optional
from peft import PeftModel
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
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


# first try
# USE_LORA = True 
# BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
# LORA_MODEL_PATH = "qwen3-1.7b-guanaco"


# second try
USE_LORA = True 
BASE_MODEL_ID   = "Qwen/Qwen1.5-4B-Chat"
LORA_MODEL_PATH = "qwen15-4b-leaf-lora"

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
    candidates: str = Field(..., description="Alternative diagnosis to consider")
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
                   shots: Optional[list[tuple[HumanMessage, AIMessage]]] = None
):
    ev = "\n".join(f"- {e}" for e in (evidence or []))

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

    evidence = RunnableLambda(lambda d: clip_topk_evidence(d["image"], k=3))
    fetch_shots    = RunnableLambda(lambda d: few_shots_collection(k=shots_k))

    prep = RunnableParallel(
        evidence=evidence,
        image=RunnableLambda(lambda d: d["image"]),
        user_input=RunnableLambda(lambda d: d["user_input"]),
        shots=fetch_shots
    )

    to_prompt = RunnableLambda(
        lambda d: build_prompt_text(d["user_input"], d["evidence"], shots=d["shots"])
    )

    chain = prep | to_prompt | llm | parser
    return chain.invoke({"image": image_path, "user_input": user_input})









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
    args = ap.parse_args()

    image_path = args.image
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