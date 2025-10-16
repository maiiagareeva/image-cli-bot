# from pipeline import image_to_data_uri
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
import base64
from pathlib import Path

def image_url(path: str) -> str:
    b = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


def few_shots_collection():
    colelction = []

    user1 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_bottom_ex1.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai1 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_bottom",
        "confidence": 0.93,
        "evidence": "On the underside of the grape leaf, two pale yellow-white fuzzy patches are visible along the veins. The texture appears powdery and slightly raised, consistent with early sporulation of downy mildew (Plasmopara viticola) that forms on the lower surface. The top surface remains mostly green and intact, indicating the infection is in its early leaf-bottom stage."
    }, ensure_ascii=False))
    colelction.append((user1, ai1))

    user2 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_bottom_ex2.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai2 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_bottom",
        "confidence": 0.91,
        "evidence": "The underside of the leaf shows diffuse, patchy white-gray fungal growth along the veins, consistent with early-stage downy mildew sporulation. The top surface is not visible but the pattern of underside lesions and powdery texture suggest infection at the lower surface in its early stage."
    }, ensure_ascii=False))
    colelction.append((user2, ai2))

    user3 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_bottom_ex3.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai3 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_top",
        "confidence": 0.88,
        "evidence": "The upper surface of the grape leaf shows faint yellowish, oil-like spots that are slightly translucent under light, typical of early-stage downy mildew on the top surface. The infection is not yet producing white fuzz below, suggesting it is in the pre-sporulation top phase of downy mildew development."
    }, ensure_ascii=False))
    colelction.append((user3, ai3))

    user4 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_top_ex1.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai4 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_top",
        "confidence": 0.92,
        "evidence": "The upper surface of the leaf exhibits numerous irregular, yellow-green oil-like lesions between veins, consistent with the early signs of downy mildew infection on the top side. The spots are diffuse and slightly translucent, indicating the beginning of chlorosis before fungal sporulation appears on the underside."
    }, ensure_ascii=False))
    colelction.append((user4, ai4))

    user5 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_top_ex2.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai5 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_top",
        "confidence": 0.94,
        "evidence": "The grape leaf surface shows several irregular yellowish 'oil spots' primarily on the upper side, located between veins and spreading outward. The tissue is slightly translucent, which is a hallmark of early downy mildew infection on the top surface before sporulation develops underneath."
    }, ensure_ascii=False))
    colelction.append((user5, ai5))

    user6 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/downy_early_leaf_top_ex3.png")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai6 = AIMessage(content=json.dumps({
        "disease": "downy_mildew_early_leaf_top",
        "confidence": 0.95,
        "evidence": "The leaf displays a mosaic of yellow angular spots between veins on the upper surface, forming a characteristic 'oil-spot' pattern typical of early downy mildew infection on the top side. The lesions align with the leaf’s vein network and lack the fuzzy white growth seen on the underside, confirming the early top-stage infection."
    }, ensure_ascii=False))
    colelction.append((user6, ai6))

    user7 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/phylloxera_ex1.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai7 = AIMessage(content=json.dumps({
        "disease": "phylloxera",
        "confidence": 0.98,
        "evidence": "The grape leaf surface is covered with numerous raised greenish galls, each corresponding to insect feeding sites caused by grape phylloxera (Daktulosphaira vitifoliae). The uniform dome-like bumps are a distinct indicator of phylloxera infestation, not fungal infection, and are primarily found on the lower leaf surface in severe cases."
    }, ensure_ascii=False))
    colelction.append((user7, ai7))

    user8 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/phylloxera_ex2.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai8 = AIMessage(content=json.dumps({
        "disease": "phylloxera",
        "confidence": 0.99,
        "evidence": "The leaf displays numerous small, dome-shaped green galls densely distributed across the surface, which are characteristic feeding-induced deformities caused by grape phylloxera. The regular, raised swellings differ from fungal lesions and are clear evidence of insect gall formation."
    }, ensure_ascii=False))
    colelction.append((user8, ai8))

    user9 = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": image_url("examples/phylloxera_ex3.jpg")}},
        {"type": "text", "text": "Task: Identify grape leaf disease. Provide JSON only."}
    ])
    ai9 = AIMessage(content=json.dumps({
        "disease": "phylloxera",
        "confidence": 0.98,
        "evidence": "The grape leaf underside shows abundant rounded protrusions—typical galls caused by phylloxera feeding. The uniform, blister-like swellings are distinct from fungal or bacterial symptoms and clearly indicate insect-induced gall formation on the lower leaf surface."
    }, ensure_ascii=False))
    colelction.append((user9, ai9))

    return colelction
