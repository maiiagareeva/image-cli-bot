import os
import base64
from pathlib import Path
from dotenv import load_dotenv, find_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from get_topk_evidence import clip_topk_evidence
from openai import OpenAI
import json

load_dotenv(find_dotenv(usecwd=True), override=True)
USE_MOCK       = os.getenv("USE_MOCK", "0") == "1"
API_KEY        = os.getenv("OPENAI_API_KEY", "")
BACKEND        = (os.getenv("LLM_BACKEND", "openai") or "openai").lower()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MODEL      = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
OPENAI_BASE_URL= (os.getenv("OPENAI_BASE_URL") or "").strip()

K=3

def cur_model():
    return OPENAI_MODEL if BACKEND=="openai" else LLM_MODEL
CUR_MODEL=cur_model();

print(f"[pipeline] USE_MOCK = {USE_MOCK} \n BACKEND={BACKEND} \n OPENAI_MODEL={OPENAI_MODEL} \n LLM_MODEL={LLM_MODEL} \n OPENAI_BASE_URL={OPENAI_BASE_URL}")

def image_to_data_uri(path: str) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

def isVLM(CUR_MODEL):
    mo=(CUR_MODEL or " ").lower()
    return ("gpt-4o" in mo) or ("vl" in mo and "qwen" in mo)

if USE_MOCK:
    class MockMode:
        def invoke(self, _msgs=None, **_kwargs):
            import json
            mock_json = {
                "disease": "early blight",
                "confidence": 0.62,
                "evidence": "mild spots on the leaf"
            }
            return AIMessage(content=json.dumps(mock_json))
    llm = MockMode()

else:
    from langchain_openai import ChatOpenAI
    if BACKEND=="openai":
        if not API_KEY or not API_KEY.startswith("sk-"):
            raise SystemExit("No valid OPENAI_API_KEY.")
    else: 
        if not OPENAI_BASE_URL:
            print("[piepeline] BACKEND =local but OPENAI_BASE_URL is not valid, please provide OPENAI_BASE_URL in env")

    kwargs={}
    if OPENAI_BASE_URL:
        kwargs["openai_api_base"]=OPENAI_BASE_URL
        kwargs["base_url"]=OPENAI_BASE_URL
        # kwargs["model_kwargs"]={"response_format": {"type": "json_object"}}

    is_openai = (BACKEND=="openai") and (not OPENAI_BASE_URL or "api.openai.com" in OPENAI_BASE_URL)

    if is_openai:
        llm = ChatOpenAI(
            model=CUR_MODEL, 
            temperature=0, 
            api_key=API_KEY,
            **kwargs,
            response_format={"type":"json_object"},
        )
    else:
        llm = ChatOpenAI(
            model=CUR_MODEL, 
            temperature=0, 
            api_key=API_KEY,
            **kwargs,
        )


def message_feed(prompt,image_path,k):
    system_guard = (
        "You are an image-vision assistant for leaf disease classification. "
        "First, analyze the image and describe its visible symptoms. "
        "Then, output a JSON object with exactly these keys:\n"
        "{\n"
        '  "disease": "<predicted disease name>",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "evidence": "<short textual explanation>"\n'
        "}\n"
        "If the leaf looks healthy, output 'Healthy Leaf' as disease. "
        "Do not leave any field empty."
    )
    clip_evidence = clip_topk_evidence(image_path, k)
    msg=[]
    if isVLM(CUR_MODEL):
        msg=[
            SystemMessage(content=system_guard),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
                {"type": "text", "text": f"{prompt}\n\n{clip_evidence}"},
            ])
        ]
    else:
        msg=[
            SystemMessage(content=system_guard),
            HumanMessage(content=[
                {"type": "text", "text": f"{prompt}\n\n{clip_evidence}"},
            ])
        ]
    return msg


# def classify_image(prompt: str, image_path: str):
#     if USE_MOCK:
#         response = llm.invoke()
#         return response.content

#     msg=message_feed(prompt,image_path,K)
#     return llm.invoke(msg).content

def classify_image(prompt: str, image_path: str):
    if USE_MOCK:
        return llm.invoke().content

    msg = message_feed(prompt, image_path, K)

    if OPENAI_BASE_URL:
        client = OpenAI(base_url=OPENAI_BASE_URL, api_key=API_KEY or "EMPTY")
    else:
        client = OpenAI(api_key=API_KEY)

    completion = client.chat.completions.create(
        model=CUR_MODEL,
        messages=[
            {"role": "system", "content": msg[0].content},
            {"role": "user", "content": msg[1].content},
        ],
    )

    print(f"[pipeline] Chat Completion response content \n {completion.model_dump_json(indent=2)} \n Chat Completion response FINISHED PRINT \n")

    return completion.choices[0].message.content

