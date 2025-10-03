import os #environment variables
import base64 #encodes image bytes into text
from pathlib import Path
from dotenv import load_dotenv, find_dotenv #put environment variable sinto a .env file (for safety)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
#HumanMessage:human input
#AIMessage: model(ai) output

#load environment variable
load_dotenv(find_dotenv(usecwd=True), override=True)
#default is 0, 0:using real API calls, 1:mock mode
USE_MOCK = os.getenv("USE_MOCK", "0") == "1"
#OpenAI's API key (we haven't have any yet)
API_KEY = os.getenv("OPENAI_API_KEY")

#for debugging, feel free to delete anything about mock if we not gonna use mock mode
print(f"[pipeline] USE_MOCK = {USE_MOCK}")

#image fiel---->data URL
#data url can be directly passed to a mutilmodel
def image_to_data_uri(path: str) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

#mock mode vs real API mode
if USE_MOCK:
    class MockMode:
        def invoke(self, _msgs=None, **_kwargs):
            import json
            #this is a simulation of a model output
            mock_json = {
                "disease": "early blight",
                "confidence": 0.62,
                "evidence": "mild spots on the leaf"
            }
            #matching response format
            return AIMessage(content=json.dumps(mock_json))
    llm = MockMode()

else:
    from langchain_openai import ChatOpenAI
    #missing or invalid API KEy
    if not API_KEY or not API_KEY.startswith("sk-"):
        raise SystemExit("No valid OPENAI_API_KEY.")
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        api_key=API_KEY,
        #enable OpenAI's JSON mode (the protocol layer forces JSON to be returned)
        model_kwargs={"response_format": {"type": "json_object"}}
        )

def classify_image(prompt: str, image_path: str) -> str:
    # Mock
    if USE_MOCK:
        response = llm.invoke()
        return response.content

    # for real api calls:
    # system gurdrails
    system_guard = (
        "You are an image-vision assistant for leaf disease classification. "
        "Return ONLY a single JSON object with EXACT keys: "
        'disease (string), confidence (number between 0 and 1), evidence (string). '
        "No extra text, no markdown, no code fences."
    )

    # multimodel
    msg = [
        SystemMessage(content=system_guard),
        HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
    ])]
    response = llm.invoke(msg)
    return response.content
