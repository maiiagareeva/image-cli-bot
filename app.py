import argparse, os, json

# from image_crop_example import center_crop_to_512
from cli.crop_send import crop_resize_512
from pipeline import classify_image, USE_MOCK
from pathlib import Path

def main():
    #confirm current mode
    print(f"[app] USE_MOCK = {USE_MOCK}")

    ap = argparse.ArgumentParser(description="Classify a leaf image with an LLM.")
    #required is Flase since we are not restric the source we get text input and image path input
    ap.add_argument("--prompt", required=False, help="...")
    ap.add_argument("--image", required=False, help="...")
    args = ap.parse_args()

    #interactive input
    if not args.prompt:
        args.prompt = input("Enter prompt text: ").strip()
    if not args.image:
        args.image = input("Enter image path: ").strip()

    if not os.path.exists(args.image):
        raise SystemExit(f"[app] Image not found: {args.image}")

    #center crop and resize to the input image
    p = Path(args.image)
    cropped = str(p.with_name(p.stem + "_512.png"))
    crop_resize_512(args.image, cropped)
    print(f"[app] Cropped → {cropped}")

    json_spec = (
        "Return ONLY a JSON object with keys: "
        "disease (string), confidence (number between 0 and 1), evidence (string). "
        "Do not include any extra commentary, markdown, or code fences."
    )
    prompt_for_model = f"{args.prompt}\n\n{json_spec}"

    #inference pipeline
    out = classify_image(prompt_for_model, cropped)
    print("\n===MODEL OUTPUT===")
    print(out)

    try:
        #parse output as JSON (structured)
        data = json.loads(out)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[app] results.json saved")

        #print out result for better in terminal
        #we can add more aspects in future if needed
        print(f"disease   : {data.get('disease')}")
        print(f"confidence: {data.get('confidence')}")
        print(f"evidence  : {data.get('evidence')}")

    except Exception as e:
        #print exception
        print(f"[app] output is not valid JSON ({type(e).__name__}). Skipped saving results.json.")

if __name__ == "__main__":
    main()