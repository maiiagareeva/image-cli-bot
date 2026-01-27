import argparse, os, json
from cli.crop_send import crop_resize_512
from pathlib import Path

def main():

    ap = argparse.ArgumentParser(description="Classify a leaf image with an VLM.")
    ap.add_argument("--prompt", required=False)
    ap.add_argument("--image", required=False)
    ap.add_argument("--backend", choices=["openai", "local"], help="override LLM_BACKEND in .env")
    ap.add_argument("--model", help="override model name (OPENAI_MODEL / LLM_MODEL)")
    ap.add_argument("--base-url", help="override OPENAI_BASE_URL (e.g., http://localhost:11434/v1)")
    ap.add_argument("--no-crop", action="store_true", help="use the original image without cropping")
    args = ap.parse_args()

    
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
        os.environ["LLM_MODEL"] = args.model
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    
    from pipeline import classify_image, USE_MOCK, BACKEND, CUR_MODEL, OPENAI_MODEL,LLM_MODEL,OPENAI_BASE_URL
    print(f"[app] USE_MOCK = {USE_MOCK} \n BACKEND={BACKEND} \n OPENAI_MODEL={OPENAI_MODEL} \n LLM_MODEL={LLM_MODEL} \n OPENAI_BASE_URL={OPENAI_BASE_URL} \n CUR_MODEL={CUR_MODEL}")

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
        "You are an image-vision assistant for leaf disease classification. "
        "First, analyze the image and briefly describe its visible symptoms. "
        "Then, output a JSON object with exactly these keys:\n"
        "{\n"
        '  "disease": "<predicted disease name>",\n'
        '  "confidence": <number between 0 and 1>,\n'
        '  "evidence": "<short textual explanation>"\n'
        "}\n"
        "If the leaf looks healthy, output 'Healthy Leaf' as disease. "
        "Do not leave any field empty."
    )
    prompt_for_model = f"{args.prompt}\n\n{json_spec}"

    out = classify_image(prompt_for_model, cropped)
    print(out)

    # try:
    #     data = json.loads(out)
    #     with open("results.json", "w", encoding="utf-8") as f:
    #         json.dump(data, f, indent=2, ensure_ascii=False)
    #     print("[app] results.json saved")


    # except Exception as e:
    #     print(f"[app] output is not valid JSON. Skipped saving results.json.")

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_path = f"results_raw_{ts}.txt"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"[app] raw saved -> {raw_path}")

    import re
    def parse_loose_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1 and e > s:
            cand = text[s:e+1]
            try:
                return json.loads(cand)
            except Exception:
                pass
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.I)
        if m:
            return json.loads(m.group(1).strip())
        raise ValueError("no-json")

    try:
        data = parse_loose_json(out)
        with open("results.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("[app] results.json saved")
    except Exception:
        print("[app] output is not valid JSON. Kept raw text only.")

if __name__ == "__main__":
    main()