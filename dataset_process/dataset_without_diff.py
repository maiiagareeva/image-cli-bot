import pandas as pd
from pathlib import Path
import json

from datasets import Dataset,DatasetDict,Features,Value,Image as HFImage

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR / "NGLD"
print("SCRIPT_DIR =", SCRIPT_DIR)
print("ROOT =", ROOT)
print("ROOT exists =", ROOT.exists())
assert ROOT.exists(), f"NGLD folder not found at {ROOT}"


rows=[]

disease_alias_map={
    "downy_early_leaf_top":"Downy Mildew",
    "downy_early_leaf_bottom":"Downy Mildew",
    "healthy":"Healthy",
}

for class_folder in ["Healthy_Leaves","Downy_Mildew"]:
    folder=ROOT/class_folder

    for image_path in folder.glob("*.jpg"):
        stem=image_path.stem
        txt_path=image_path.with_suffix(".txt")
        teacher_path=image_path.with_name(stem+".teacher.json")

        if not txt_path.exists():
            continue
        if not teacher_path.exists():
            continue

        rows.append({
            "image_id":stem,
            "class_folder":class_folder,
            "image_path":str(image_path),
            "txt_path":str(txt_path),
            "teacher_path":str(teacher_path),
        })
images_DF=pd.DataFrame(rows)

print(images_DF.head(10))
print(len(images_DF))

samples=[]
for _, r in images_DF.iterrows():
    image_id = r["image_id"]
    image_path = r["image_path"]

    teacher = json.loads(Path(r["teacher_path"]).read_text(encoding="utf-8"))

    raw_disease = teacher.get("disease", "").strip().lower()

    if raw_disease not in disease_alias_map:
        continue

    teacher["disease"] = disease_alias_map[raw_disease]
    pred = teacher["disease"]

    expected = r["class_folder"]

    if expected == "Healthy Leaves" and pred != "Healthy":
        continue
    if expected == "Downy Mildew" and pred != "Downy Mildew":
        continue

    teacher.pop("differentials", None)
    teacher.pop("references", None)
    teacher.pop("confidence", None)
    teacher.pop("severity", None)

    samples.append({
        "sample_id": f"{image_id}|teacher",
        "image_id": image_id,
        "image_path": image_path,
        "task": "teacher_json",
        "prompt": (
            "You are a grape leaf disease diagnosis assistant. "
            "Analyze the image and output a diagnosis in the following JSON schema."
        ),
        "answer": json.dumps(
            teacher,
            ensure_ascii=False,
            indent=2,
            sort_keys=False
        ),
    })


    
samples_DF=pd.DataFrame(samples)

print(samples_DF.head(10))
print(len(samples_DF))


images_DF=images_DF.sample(frac=1.0,random_state=42).reset_index(drop=True)
n=len(images_DF)
n_train=int(0.8*n)
n_val=int(0.1*n)

images_DF["split"]="test"
images_DF.loc[:n_train-1,"split"]="train"
images_DF.loc[n_train:n_train+n_val-1,"split"]="val"

samples_DF=samples_DF.merge(images_DF[["image_id","split"]],on="image_id",how="left")

print(samples_DF["split"].value_counts())


OUT = Path("dataset_ngld_qwen") / "annotations"
OUT.mkdir(parents=True,exist_ok=True)

for split in ["train","val","test"]:
    part=samples_DF[samples_DF["split"]==split]
    out_path=OUT/f"{split}.jsonl"
    part[["sample_id","image_path","task","prompt","answer"]].to_json(
        out_path,orient="records",lines=True,force_ascii=False
    )

Path("dataset_ngld_qwen").mkdir(exist_ok=True)
samples_DF.to_parquet("dataset_ngld_qwen/samples.parquet",index=False)
images_DF.to_parquet("dataset_ngld_qwen/images.parquet",index=False)





samples_DF["image"]=samples_DF["image_path"].apply(lambda p:Path(p).as_posix())

df=samples_DF.merge(
    images_DF[["image_id","class_folder","split"]],
    on="image_id",
    how="left"
)

if "split" not in df.columns:
    if "split_x" in df.columns:
        df["split"] = df["split_x"]
    elif "split_y" in df.columns:
        df["split"] = df["split_y"]

df = df[df["image"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

df = df[["sample_id", "image_id", "class_folder", "split", "task", "prompt", "answer", "image"]].copy()

features = Features({
    "sample_id": Value("string"),
    "image_id": Value("string"),
    "class_folder": Value("string"),
    "split": Value("string"),
    "task": Value("string"),
    "prompt": Value("string"),
    "answer": Value("string"),
    "image": HFImage(),
})

def make_split(df_part):
    df_part = df_part[["sample_id","image_id","class_folder","task","prompt","answer","image"]].copy()
    return Dataset.from_pandas(df_part, features=features, preserve_index=False)

ds_train = make_split(df[df["split"] == "train"])
ds_val   = make_split(df[df["split"] == "val"])
ds_test  = make_split(df[df["split"] == "test"])

ds_dict = DatasetDict({
    "train": ds_train,
    "val": ds_val,
    "test": ds_test
})

ds_dict.push_to_hub("qingwuuu/ngld-grape-leaf-vlm-w-img-without-diff-ref", private=False)
