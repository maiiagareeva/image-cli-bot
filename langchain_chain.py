import os, base64, json, pathlib
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from get_topk_evidence import clip_topk_evidence
from few_shots import few_shots_collection
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=False)
BACKEND=os.getenv("LLM_BACKEND", "openai").lower()
MODEL=os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"

if BACKEND=="openai":
    BASE_URL=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    API_KEY=os.getenv("OPENAI_API_KEY")
    assert API_KEY, "OPENAI_API_KEY missing"
elif BACKEND=="local":
    BASE_URL=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
    API_KEY=os.getenv("OPENAI_API_KEY", "sk-local-dummy")


llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1,
    max_tokens=4000,
)


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


def img_to_data_uri(path: str):
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_messages(image_path: str, 
                   prompt: str,
                   evidence: Optional[List[str]] = None,
                   shots: Optional[list[tuple[HumanMessage, AIMessage]]] = None):
    ev = "\n".join(f"- {e}" for e in (evidence or []))
    sys = SystemMessage(content=(
        "You are an expert grape-leaf pathologist.\n"
        "Return STRICTLY a single JSON object conforming to the given schema "
        "(no preface, no markdown, no extra text). "
        "Be precise and concrete. Avoid speculation beyond visible cues."
    ))

    schema_hint = (
        "JSON fields:\n"
        "  disease (string), confidence (0-1 float), severity (mild|moderate|severe),\n"
        "  indicators (>=4 short bullet strings), regions (optional list),\n"
        "  differentials (2-3 objects: {candidates, reason_less_likely}),\n"
        "  recommended_checks (>=3 items),\n"
        "  evidence (>=120 words, one paragraph, objective visual rationale),\n"
        "  references (optional short factual bullets).\n"
    )

    human_text = (
        f"{prompt}\n\n"
        f"{schema_hint}"
        "Use only information visible in the image plus the bullets below.\n"
        "Top-K visual evidence:\n"
        f"{ev}\n\n"
        "Output: JSON ONLY."
    )

    human_content = [
        {"type": "text", "text": human_text},
        {"type": "image_url", "image_url": {"url": img_to_data_uri(image_path)}},
    ]

    msgs = [sys]
    if shots:
        for h, a in shots:
            msgs.append(h)
            msgs.append(a)
    msgs.append(HumanMessage(content=human_content))
    return msgs


def langchain_run(image_path: str,
              prompt: str = "Identify the grape leaf disease.",
              shots_k: int = 4):

    evidence = RunnableLambda(lambda d: clip_topk_evidence(d["image"], k=3))
    fetch_shots    = RunnableLambda(lambda d: few_shots_collection(k=shots_k))

    prep = RunnableParallel(
        evidence=evidence,
        image=RunnableLambda(lambda d: d["image"]),
        prompt=RunnableLambda(lambda d: d["prompt"]),
        shots=fetch_shots
    )

    to_messages = RunnableLambda(
        lambda d: build_messages(d["image"], d["prompt"], d["evidence"], shots=d["shots"])
    )

    chain = prep | to_messages | llm | parser
    return chain.invoke({"image": image_path, "prompt": prompt})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Identify the grape leaf disease.")
    args = ap.parse_args()

    res = langchain_run(args.image, args.prompt)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    pathlib.Path("langchain_with_ICL_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
