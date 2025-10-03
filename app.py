import argparse, os, json
from cli.crop_send import crop_resize_512
from pipeline import classify_image, USE_MOCK
from pathlib import Path

def main():
    print(f"[app] USE_MOCK = {USE_MOCK}")

    ap = argparse.ArgumentParser(description="Classify a leaf image with an VLM.")
    ap.add_argument("--prompt", required=False)
    ap.add_argument("--image", required=False)
    args = ap.parse_args()

    if not args.prompt:
        args.prompt = input("Enter prompt text: ").strip()
    if not args.image:
        args.image = input("Enter image path: ").strip()

    if not os.path.exists(args.image):
        raise SystemExit(f"[app] Image not found: {args.image}")

    p = Path(args.image)
    cropped = str(p.with_name(p.stem + "_512.png"))
    crop_resize_512(args.image, cropped)
    print(f"[app] Cropped → {cropped}")

    json_spec = (
        #TODO: learn more about Gopher Eye's detection, add more constrained
        "return your answer with following aspects in JSON format: "
        "disease (string), confidence (number between 0 and 1), evidence (string). "
    )
    prompt_for_model = f"{args.prompt}\n\n{json_spec}"

    out = classify_image(prompt_for_model, cropped)
    print(out)

    try:
        data = json.loads(out)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[app] results.json saved")


    except Exception as e:
        print(f"[app] output is not valid JSON. Skipped saving results.json.")

if __name__ == "__main__":
    main()