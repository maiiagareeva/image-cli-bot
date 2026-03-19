import json
import shutil
import hashlib
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "NGLD"
assert ROOT.exists(), f"NGLD folder not found at {ROOT}"

OUT_ROOT = Path("dataset_ngld_qwen")
OUT_ANN = OUT_ROOT / "annotations"
OUT_IMG = OUT_ROOT / "images"
OUT_ANN.mkdir(parents=True, exist_ok=True)
OUT_IMG.mkdir(parents=True, exist_ok=True)

folder_to_label = {
    "Healthy_Leaves": "Healthy",
    "Downy_Mildew": "Downy Mildew",
}

disease_alias_map = {
    "downy_early_leaf_top": "Downy Mildew",
    "downy_early_leaf_bottom": "Downy Mildew",
    "downy mildew": "Downy Mildew",
    "healthy": "Healthy",
}

def file_sha1(p: Path, chunk_size=1024 * 1024) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

rows = []
for class_folder in ["Healthy_Leaves", "Downy_Mildew"]:
    folder = ROOT / class_folder
    if not folder.exists():
        continue

    for image_path in folder.glob("*.jpg"):
        stem = image_path.stem
        txt_path = image_path.with_suffix(".txt")
        teacher_path = image_path.with_name(stem + ".teacher.json")

        if not txt_path.exists():
            continue
        if not teacher_path.exists():
            continue

        uid = file_sha1(image_path)[:24]

        new_image_path = OUT_IMG / f"{uid}.jpg"
        if not new_image_path.exists():
            shutil.copy2(image_path, new_image_path)

        rows.append({
            "uid": uid,
            "expected_label": folder_to_label[class_folder],
            "image": new_image_path.as_posix(),
            "teacher_path": str(teacher_path),
        })

images_DF = pd.DataFrame(rows)
print("images_DF size =", len(images_DF))
print(images_DF.head(3))

samples = []
for _, r in images_DF.iterrows():
    uid = r["uid"]
    teacher = json.loads(Path(r["teacher_path"]).read_text(encoding="utf-8"))

    raw = (teacher.get("disease", "") or "").strip().lower()
    if raw not in disease_alias_map:
        continue

    teacher["disease"] = disease_alias_map[raw]
    pred_label = teacher["disease"]
    expected_label = r["expected_label"]

    if pred_label != expected_label:
        continue

    teacher.pop("differentials", None)
    teacher.pop("references", None)
    teacher.pop("confidence", None)
    teacher.pop("severity", None)
    teacher.pop("regions", None)

    samples.append({
        "sample_id": f"{uid}|teacher",
        "image_id": uid,
        "task": "teacher_json",
        "prompt": (
            "You are a grape leaf disease diagnosis assistant. "
            "Analyze the image and output a diagnosis in the following JSON schema."
        ),
        "answer": json.dumps(teacher, ensure_ascii=False, indent=2, sort_keys=False),
        "image": r["image"],
    })

samples_DF = pd.DataFrame(samples)
print("samples_DF size =", len(samples_DF))
print(samples_DF.head(3))

images_DF = images_DF[["uid"]].drop_duplicates().sample(frac=1.0, random_state=42).reset_index(drop=True)
n = len(images_DF)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

images_DF["split"] = "test"
images_DF.loc[:n_train - 1, "split"] = "train"
images_DF.loc[n_train:n_train + n_val - 1, "split"] = "val"

samples_DF = samples_DF.merge(images_DF, left_on="image_id", right_on="uid", how="left")
samples_DF.drop(columns=["uid"], inplace=True)

print(samples_DF["split"].value_counts(dropna=False))

for split in ["train", "val", "test"]:
    part = samples_DF[samples_DF["split"] == split]
    out_path = OUT_ANN / f"{split}.jsonl"
    part[["sample_id", "image", "task", "prompt", "answer"]].to_json(
        out_path, orient="records", lines=True, force_ascii=False
    )

OUT_ROOT.mkdir(exist_ok=True)
samples_DF.to_parquet(OUT_ROOT / "samples.parquet", index=False)
images_DF.to_parquet(OUT_ROOT / "images.parquet", index=False)

features = Features({
    "sample_id": Value("string"),
    "image_id": Value("string"),
    "split": Value("string"),
    "task": Value("string"),
    "prompt": Value("string"),
    "answer": Value("string"),
    "image": HFImage(),
})

def make_split(df_part: pd.DataFrame) -> Dataset:
    df_part = df_part[["sample_id", "image_id", "split", "task", "prompt", "answer", "image"]].copy()
    return Dataset.from_pandas(df_part, features=features, preserve_index=False)

ds_dict = DatasetDict({
    "train": make_split(samples_DF[samples_DF["split"] == "train"]),
    "val": make_split(samples_DF[samples_DF["split"] == "val"]),
    "test": make_split(samples_DF[samples_DF["split"] == "test"]),
})

ds_dict.push_to_hub("qingwuuu/ngld-grape-leaf-vlm-w-img-without-diff-ref", private=False)
