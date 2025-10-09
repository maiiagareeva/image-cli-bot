import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from crop_send import crop_resize_512
from pipeline import classify_image, USE_MOCK, API_KEY

def run_chat():
    print(f"[chat_bot] USE_MOCK = {USE_MOCK}")
    print("Type 'exit' to quit.\n")

    last_image = None

    while True:
        # prompt
        prompt = input("Enter your prompt: ").strip()
        if prompt.lower() in ["exit", "quit"]:
            print("[chat_bot] Goodbye!")
            break

        # ask for image path
        image_path = input("Enter image path: ").strip()
        if image_path == "":
            image_path = last_image
        if not Path(image_path).exists():
            print("[chat_bot] Invalid image path. Try again.")
            continue
        last_image = image_path

        # make the image cropped
        p = Path(image_path)
        cropped = str(p.with_name(p.stem + "_512.png"))
        crop_resize_512(image_path, cropped)
        print(f"[chat_bot] Cropped → {cropped}")

        # inference
        out = classify_image(prompt, cropped)
        print("\n=== MODEL OUTPUT ===")
        print(out)

        # saving structured results in JSON format
        try:
            data = json.loads(out)
            with open("results.json", "w", encoding="utf-8") as f:
                json.dump(data, f)
            print("[chat_bot] results.json saved")
        except Exception:
            print("[chat_bot] Output is not valid JSON. Skipped saving.")

if __name__ == "__main__":
    run_chat()