import csv,re,unicodedata
import random
from pathlib import Path
import yaml
import json
#change database path to a new path once preprocess is done
database=Path("leaf_disease_vlm")
CLASS_yaml="classes.yaml"
store= Path("metadata_manifest.csv")    

num_test = 3
num_val  = 2
image_format = {".jpg",".jpeg",".png"}

def read_texts(txt_path: Path):
    if not txt_path.exists():
        return []
    sentences = txt_path.read_text(encoding="utf-8").splitlines()
    all = []
    for sen in sentences:
        sen = unicodedata.normalize("NFKC", sen)
        sen = re.sub(r"\s+", " ", sen).strip()
        all.append(sen)
    return all

def find_image(cdir: Path, stem: str):
    for format in image_format:
        path = cdir / f"{stem}{format}"
        if path.exists(): 
            return path
    return None

def main():
    with open(CLASS_yaml,"r",encoding="utf-8") as f:
        data=yaml.safe_load(f)
    classes = data.get("CLASS",[])
    classes.sort()
    rows = []

    for cls in classes:
        cdir = database / cls
        #001,002,003,,,,,,
        stems = set()
        for p in cdir.iterdir():
            if p.is_file():
                stems.add(p.stem)
        stems = sorted(stems)

        per_class = []
        for stem in stems:
            img = find_image(cdir, stem)
            if img is None:
                continue
            txt = cdir / f"{stem}.txt"
            texts = read_texts(txt)
            # json_txt = json.dumps(texts, ensure_ascii=False)

            per_class.append({
                "id": f"{cls}/{stem}",
                "image_path": str(img.as_posix()),
                "class_name": cls,
                "texts": "|".join(texts),
                # "texts": json_txt,
                "split": ""
            })

        random.shuffle(per_class)
        test = per_class[:num_test]
        val  = per_class[num_test:num_test+num_val]
        train= per_class[num_test+num_val:]
        for row in test:  
            row["split"] = "test"
        for row in val:   
            row["split"] = "val"
        for row in train: 
            row["split"] = "train"
        rows.extend(per_class)

    store.parent.mkdir(parents=True, exist_ok=True)
    with store.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","image_path","class_name","texts","split"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote_to_file_path={store}                                    total_data_pairs={len(rows)}                                               num_of_classes={classes}")

if __name__ == "__main__":
    main()
