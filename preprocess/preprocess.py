import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import nltk
import clip

import pandas as pd
from PIL import Image
from tqdm import tqdm
import unicodedata
from common_words import DOMAIN_STOPWORDS

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from cli.crop_send import crop_resize_512


IMG_EXTS = {".jpg", ".jpeg", ".png"}
nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
STOPWORDS |= DOMAIN_STOPWORDS
tokenizer = clip.tokenize

def shorten_to_tokens_max(text: str, tokenizer, max_tokens: int = 77) -> str:
    try:
        tokens = tokenizer(text, truncate=False)
        if tokens.shape[1] <= max_tokens:
            return text
    except RuntimeError:
        pass

    words = text.split()
    while words:
        attempt = " ".join(words)
        try:
            tokens = tokenizer(attempt, truncate=False)
            if tokens.shape[1] <= max_tokens:
                return attempt
        except RuntimeError:
            pass
        words.pop()
    return ""


def normalize_text(s: str, max_len: int = 77) -> str:
    # normalization
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\u00A0", " ").strip().lower()  # non-breaking spaces turn into space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    all_words = [w for w in s.split() if w not in STOPWORDS] # creating a new filtered text from the old one
    result = " ".join(all_words)
    try:
        tokens = tokenizer(result, truncate=False)
    except RuntimeError:
        result = shorten_to_tokens_max(result, tokenizer, max_tokens=max_len)
        tokens = tokenizer(result, truncate=False)

    while tokens.shape[1] > max_len:
        result = shorten_to_tokens_max(result, tokenizer, max_tokens=max_len)
        tokens = tokenizer(result, truncate=False)
    return result


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def collect_classes(root: Path) -> List[Path]:
    # subclusses with imgaes inside
    classes = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            # Consider it a class if it contains at least one !image file! anywhere within.
            has_image = any(is_image_file(p) for p in child.rglob("*"))
            if has_image:
                classes.append(child)
    return classes


def write_class_map_yaml(class_dirs: List[Path], out_path: Path) -> None:
    # Write a simple YAML with a list of class names of folder name.
    class_names = [c.name for c in class_dirs]
    content_lines = ["classes:"]
    for name in class_names:
        content_lines.append(f"  - {name}")
    out_path.write_text("\n".join(content_lines), encoding="utf-8")


def pair_records_for_class_dir(class_dir: Path, root: Path) -> Tuple[List[Dict], List[str], List[str]]:
    """
    For one class folder it return:
        records: list of {"image_path": <rel path>, "text": <raw>, "class": <class_name>}
        missing_txt: list of relative image paths with no matching .txt
        img_missing: list of relative text paths with no matching image
    """
    records = []
    missing_txt = []
    img_missing = []

    images_by_stem: Dict[str, Path] = {}
    txts_by_stem: Dict[str, Path] = {}

    for p in class_dir.iterdir():
        if p.is_file():
            stem = p.stem
            if is_image_file(p):
                images_by_stem[stem] = p
            elif p.suffix.lower() == ".txt":
                txts_by_stem[stem] = p

    # find/create pairs if don't
    for stem, img_p in images_by_stem.items():
        rel_img = img_p.relative_to(root).as_posix()
        if stem in txts_by_stem:
            txt_p = txts_by_stem[stem]
            try:
                text = txt_p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                text = ""
            records.append(
                {"image_path": rel_img, "text": text, "class": class_dir.name}
            )
        else:
            missing_txt.append(rel_img)

    # (no image)
    for stem, txt_p in txts_by_stem.items():
        if stem not in images_by_stem:
            img_missing.append(txt_p.relative_to(root).as_posix()) # convert to UNIX-style
    return records, missing_txt, img_missing


def resize_and_save(src_path: Path, out_path: Path, size: int = 512) -> None:
    """
    call cli/crop_send.crop_resize_512().
    Resizes and crops image to default 512x512 and saves to out_path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop_resize_512(str(src_path), str(out_path))


def main():
    parser = argparse.ArgumentParser(description="Preprocess paired image/txt dataset for CLIP.")
    parser.add_argument("--root", required=True, help="Root folder (e.g., leaf_disease_vlm).")
    parser.add_argument("--manifest-out", default="metadata_manifest.csv", help="Where to write raw manifest CSV.")
    parser.add_argument("--clean-out", default="metadata_clean.csv", help="Where to write cleaned manifest CSV.")
    parser.add_argument("--result-out", default="preprocess_result.json", help="Where to write JSON result.")
    parser.add_argument("--output-images-dir", default=None, help="If set, write resized RGB images mirroring structure here.")
    parser.add_argument("--resize", type=int, default=None, help="If set, resize images to NxN (e.g., 224).")
    parser.add_argument("--min-text-len", type=int, default=1, help="Drop rows whose cleaned text is shorter than this.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    # Collecting class folders (collect_classes function above)
    class_dirs = collect_classes(root)
    if not class_dirs:
        raise SystemExit(f"No class folders with images found in {root}.")

    # Creating class_map.yaml
    class_map_path = Path("class_map.yaml")
    write_class_map_yaml(class_dirs, class_map_path) # to simple yaml file

    # Pairing records per class
    all_records: List[Dict] = []
    all_missing_txt: List[str] = []
    all_img_missing: List[str] = []

    for cdir in class_dirs:
        records, missing_txt, img_missing = pair_records_for_class_dir(cdir, root)
        all_records.extend(records)
        all_missing_txt.extend(missing_txt)
        all_img_missing.extend(img_missing)

    if not all_records:
        raise SystemExit("No valid image+txt pairs were found in the class.")

    # taking initial manifest
    df = pd.read_csv(args.manifest_out)

    # Cleaning text and filtering short ones
    df_clean = df.copy()
    df_clean["texts"] = df_clean["texts"].apply(normalize_text) # text clean utf
    if args.min_text_len > 1:
        df_clean = df_clean[df_clean["texts"].str.len() >= args.min_text_len].reset_index(drop=True)
    df_clean.to_csv(args.clean_out, index=False, encoding="utf-8")

    # resizing the image in db
    resized_count = 0
    output_images_dir = Path(args.output_images_dir) if args.output_images_dir else None
    if args.resize and output_images_dir:
        for _, row in tqdm(df_clean.iterrows(), total=len(df_clean)): # loop and visible progress bar
            rel_path = str(row["image_path"])
            if str(Path(rel_path)).startswith("leaf_disease_vlm/"):
                rel_path = Path(rel_path).relative_to("leaf_disease_vlm")

            src = root / Path(rel_path)
            dst = output_images_dir / Path(rel_path)
            resize_and_save(src, dst, args.resize)
            resized_count += 1

            text_content = str(row["texts"])
            txt_path = dst.with_suffix(".txt")
            txt_path.write_text(text_content, encoding="utf-8")


    # result what was done
    result = {
        "root": str(root),
        "num_classes": len(class_dirs),
        "classes": [c.name for c in class_dirs],
        "pairs_in_manifest": int(df.shape[0]),
        "pairs_after_cleaning": int(df_clean.shape[0]),
        "missing_txt": all_missing_txt,
        "img_missing": all_img_missing,
        "missing_txt_count": len(all_missing_txt),
        "img_missing_count": len(all_img_missing),
        "resized_images_count": resized_count,
        "manifest_out": str(Path(args.manifest_out).resolve()),
        "clean_out": str(Path(args.clean_out).resolve()),
        "class_map_yaml": str(class_map_path.resolve()),
        "output_images_dir": str(Path(args.output_images_dir).resolve()) if args.output_images_dir else None,
    }
    Path(args.result_out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Console summary
    print("\n=== Preprocess Summary ===")
    print(json.dumps(result))
    print("\nFiles written:")
    print(f" - Manifest: {result['manifest_out']}")
    print(f" - Cleaned manifest: {result['clean_out']}")
    print(f" - Class map: {result['class_map_yaml']}")
    if result['output_images_dir']:
        print(f" - Resized images dir: {result['output_images_dir']}")
    print(f" - result: {Path(args.result_out).resolve()}")
    print("\n Done.")
    

if __name__ == "__main__":
    main()
