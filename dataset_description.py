import os
import csv
import base64
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

ROOT_DIR = "Niphad Grape Leaf Disease Dataset (NGLD)"
OUTPUT_CSV = "grape_leaf_descriptions.csv"

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
    "4) Background & context: type of background (plain, soil, other leaves, sky, lab surface), and whether "
    "other plant parts or objects are visible.\n"
    "5) Image quality: note any blur, shadows, over/under-exposure, strong reflections, or occlusions.\n\n"
    "Important constraints:\n"
    "- Describe ONLY what is visually observable; do NOT mention or guess the disease name or label.\n"
    "- Start with a phrase like 'Close-up photo of a grape leaf...' or similar.\n"
    "- Even if the image is slightly blurred or low resolution, still describe approximate colors, shapes, "
    "and patterns instead of saying that nothing is visible."
)


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# def generate_description(image_b64):
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": VISION_PROMPT},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{image_b64}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             max_tokens=220,
#             temperature=0.1,
#             top_p=0.9
#         )

#         content = response.choices[0].message.content

#         if not content or content.strip() == "":
#             return "[EMPTY RESPONSE]"

#         text = content.strip()
#         text = " ".join(text.split())
#         return text


#     except Exception as e:
#         return f"[ERROR] {e}"

def generate_description(image_b64):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": VISION_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this grape leaf image according to the instructions."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=220,
            temperature=0.1,
            top_p=0.9,
        )

        content = response.choices[0].message.content
        if not content or content.strip() == "":
            return "[EMPTY RESPONSE]"

        text = content.strip()
        text = " ".join(text.split())
        return text

    except Exception as e:
        return f"[ERROR] {e}"


def main():
    rows = []
    print(f"Scanning dataset in {ROOT_DIR} ...")

    for category in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category)
        if not os.path.isdir(category_path):
            continue

        print(f"\nProcessing category: {category}")

        for filename in tqdm(os.listdir(category_path)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(category_path, filename)
            img_b64 = encode_image(img_path)
            desc = generate_description(img_b64)

            rows.append([filename, category, desc])

    print(f"\nSaving to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "description"])
        writer.writerows(rows)

    print("Finished!")


if __name__ == "__main__":
    main()
