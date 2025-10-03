import os
import base64
from pathlib import Path
from dotenv import load_dotenv, find_dotenv 
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(find_dotenv(usecwd=True), override=True)
USE_MOCK = os.getenv("USE_MOCK", "0") == "1"
API_KEY = os.getenv("OPENAI_API_KEY")

print(f"[pipeline] USE_MOCK = {USE_MOCK}")

def image_to_data_uri(path: str) -> str:
    # TODO: considering handle more image structure, desigend to process png, but jpg works as well during test
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

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
    #TODO:do a safer and a more precise check
    if not API_KEY or not API_KEY.startswith("sk-"):
        raise SystemExit("No valid OPENAI_API_KEY.")
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        api_key=API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}}
        )

def classify_image(prompt: str, image_path: str) -> str:
    if USE_MOCK:
        response = llm.invoke()
        return response.content

    system_guard = (
        #TODO: learn more about Gopher Eye's detection, add more constrained
        "You are an image-vision assistant for leaf disease classification. "
        "return your answer with following aspects in JSON format: "
        'disease (string), confidence (number between 0 and 1), evidence (string). '
    )

    msg = [
        SystemMessage(content=system_guard),
        HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
    ])]
    response = llm.invoke(msg)
    return response.content
