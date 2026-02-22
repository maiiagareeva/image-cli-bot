import os
import csv
import json
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

DATA_ROOT = Path("GreenHouse_img")
# IMAGE_DIRS = [DATA_ROOT / "images1", DATA_ROOT / "images2"]
# IMAGE_DIRS = [DATA_ROOT / "image_test_feb_18"]
IMAGE_DIRS = [DATA_ROOT / "next118"]
LABEL_DIR = DATA_ROOT / "labels"

OUTPUT_CSV = "2026_Feb_grape_leaf_structured_reports_test.csv"

ID_TO_DISEASE = {
    0: "Healthy",
    1: "Downy Mildew",
    2: "Powdery Mildew",
}

VISION_PROMPT = (
    "You are an expert plant pathologist describing grape leaf disease images. "
    "Your descriptions will be used as training captions for a vision model (LoRA fine-tuning).\n\n"
    "Task: Given a single grape leaf image, output ONE detailed English description in 2–4 sentences "
    "(around 80–150 words).\n\n"
    "Always cover ALL of the following aspects explicitly:\n"
    "1) Leaf: overall shape (round, lobed, etc.), apparent size, main color tones, venation pattern, "
    "and surface texture (smooth, glossy, leathery, wrinkled, etc.).\n"
    "2) Lesions / symptoms: type (spots, patches, blight, mildew, holes), color, shape, border sharpness, "
    "distribution (scattered, clustered, along veins or margin), and an approximate percentage of leaf area affected.\n"
    "3) Physiological changes: chlorosis (yellowing), necrosis (dead tissue), curling, deformation, wilting, "
    "or perforations if visible.\n"
    "Important constraints:\n"
    "- Describe ONLY what is visually observable; do NOT mention or guess the disease name or label.\n"
    "- Start with a phrase like 'Close-up photo of a grape leaf...' or similar.\n"
    "- Even if the image is slightly blurred or low resolution, still describe approximate colors, shapes, "
    "and patterns instead of saying that nothing is visible."
)

STRUCTURED_SCHEMA = {
    "name": "grape_leaf_report",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "disease_class": {"type": "string", "description": "The categorical label provided."},
            "leaf_morphology": {
                "type": "object",
                "properties": {
                    "shape": {"type": "string"},
                    "texture": {"type": "string"},
                    "color_tones": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["shape", "texture", "color_tones"]
            },
            "indicators": {
                "type": "array",
                "description": "Granular visual symptoms. Provide at least 8 specific observations.",
                "items": {"type": "string"},
                "minItems": 8,
                "maxItems": 16
            },
            "recommended_checks": {
                "type": "array",
                "description": "3-5 physical inspection steps (e.g., moisture check, abaxial inspection).",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 6
            },
            "evidence": {
                "type": "string", 
                "description": "A technical description (150-200 words). DO NOT use disease names or treatment advice.",
                "minLength": 500 
            }
        },
        "required": ["disease_class", "leaf_morphology", "indicators", "recommended_checks", "evidence"]
    }
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

HEALTHY_CLAIMS = [
    "appears healthy",
    "looks healthy",
    "no visible lesions",
    "no visible signs of disease",
    "no discoloration",
    "no visible discoloration",
    "no signs of infection",
]

def contradicts_label(evidence: str, disease_label: str) -> bool:
    e = (evidence or "").lower()
    if disease_label != "Healthy":
        return any(p in e for p in HEALTHY_CLAIMS)
    return False

def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def read_label_for_stem(stem: str) -> str | None:
    label_path = LABEL_DIR / f"{stem}.txt"
    if not label_path.exists():
        return None
    raw = label_path.read_text(encoding="utf-8").strip()
    if raw == "":
        return None
    try:
        idx = int(raw)
    except ValueError:
        return None
    return ID_TO_DISEASE.get(idx)


def generate_structured_report(image_b64: str, disease_label: str) -> dict:
    # revised Prompt
    PROMPT_SYSTEM = (
        "You are a professional phytopathologist creating training data for a VLM. "
        "Your goal is to describe ONLY the visual pixels. "
        "STRICT RULE: Do NOT mention the disease name (e.g., 'Downy Mildew') or specific "
        "pathogens (e.g., 'Plasmopara viticola') in the 'evidence' or 'indicators' fields. "
        "Focus on: morphology, chlorotic patterns, and fungal structures."
    )
    
    USER_CONTENT = (
        f"This specimen is classified as '{disease_label}'.\n\n"
        "REQUIRED DETAIL:\n"
        "1. INDICATORS: Provide at least 8 distinct technical indicators focusing on "
        "spatial distribution (base vs. tip) and venation interaction (vein-bounded vs. crossing).\n"
        "2. EVIDENCE: Write a 150+ word technical description. Describe the visual symptoms "
        "without naming the disease. Use terms like 'adaxial/abaxial', 'interveinal', and 'necrotic'.\n"
        "3. RECOMMENDED CHECKS: Suggest 3-5 physical actions a farmer should take to confirm "
        "symptoms not visible in this specific photo angle."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_CONTENT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]}
            ],
            response_format={"type": "json_schema", "json_schema": STRUCTURED_SCHEMA},
            temperature=0.3, # Lowered for consistency
            max_tokens=1000  # Increased for detailed evidence
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            return {"disease": disease_label, "indicators": [], "recommended_checks": [], "evidence": "[EMPTY RESPONSE]"}

        data = json.loads(content)

        data["disease"] = disease_label

        # if contradiction -> retry once, stricter
        if contradicts_label(data.get("evidence", ""), disease_label):
            response2 = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": VISION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": (
                                "Your previous evidence contradicts the ground-truth label.\n"
                                f"Ground-truth label: {disease_label}\n"
                                "Rewrite the JSON so the evidence and indicators are consistent with this label.\n"
                                "Do NOT say the leaf is healthy or that there are no visible symptoms.\n"
                                "If symptoms are subtle, say they are subtle/hard to see and focus on any small color/texture cues.\n"
                                "Keep indicators as symptom cues (not leaf shape/venation).\n"
                            )},
                            {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{image_b64}"}}
                        ]
                    }
                ],
                response_format={"type": "json_schema", "json_schema": STRUCTURED_SCHEMA},
                temperature=0.0,
                top_p=1.0,
                max_tokens=650
            )
            data = json.loads(response2.choices[0].message.content)
            data["disease"] = disease_label

        if not isinstance(data.get("indicators"), list):
            data["indicators"] = []
        if not isinstance(data.get("recommended_checks"), list):
            data["recommended_checks"] = []
        if not isinstance(data.get("evidence"), str):
            data["evidence"] = ""

        return data

    except Exception as e:
        return {"disease": disease_label, "indicators": [], "recommended_checks": [], "evidence": f"[ERROR] {e}"}

def save_json_report_next_to_image(img_path: Path, report: dict) -> Path:
    json_path = img_path.with_suffix(".json")
    tmp_path = json_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(json_path)
    return json_path

def iter_all_images():
    for d in IMAGE_DIRS:
        if not d.exists():
            continue
        split_name = d.name
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p, split_name, p.name, p.stem

def main():
    rows = []
    images = list(iter_all_images())
    print(f"Found {len(images)} images across: {[str(d) for d in IMAGE_DIRS]}")

    missing_label = 0
    for img_path, split_name, filename, stem in tqdm(images):
        print("img_path, split_name, filename, stem\n", img_path, split_name, filename, stem)
        disease = read_label_for_stem(stem)
        if disease is None:
            missing_label += 1
            continue

        img_b64 = encode_image(img_path)
        report = generate_structured_report(img_b64, disease)

        json_path = save_json_report_next_to_image(img_path, report)

        rows.append([
            split_name,
            str(img_path.relative_to(DATA_ROOT)),
            filename,
            stem,
            disease,
            str(json_path.relative_to(DATA_ROOT)),
            "; ".join(report.get("indicators", [])),
            "; ".join(report.get("recommended_checks", [])),
            report.get("evidence", "")
        ])

    print(f"Missing label files or unreadable labels: {missing_label}")

    print(f"\nSaving CSV to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "relative_path",
            "filename",
            "id_stem",
            "disease",
            "json_path",
            "indicators_joined",
            "recommended_checks_joined",
            "evidence"
        ])
        writer.writerows(rows)

    print(f"Finished! Wrote {len(rows)} rows and {len(rows)} JSON files.")

if __name__ == "__main__":
    main()
