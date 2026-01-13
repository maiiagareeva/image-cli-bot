import json
import random
from pathlib import Path

def load_fewshot_pool(root_dir):
    pool=[]

    root=Path(root_dir)
    for cls_dir in root.iterdir():
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*.teacher.json"):
            with open(p,"r") as f:
                sample=json.load(f)

            pool.append(
                {
                    "class":cls_dir.name,
                    "data":sample
                }
            )

    return pool

def select_fewshots(pool,k):
    by_class={}
    for item in pool:
        by_class.setdefault(item["class"],[]).append(item["data"])

    classes=list(by_class)
    random.shuffle(classes)

    selected=[]
    for cls in classes:
        selected.append(by_class[cls][0])
        if len(selected)>=k:
            break

    return selected


def format_fewshots(fewshots):
    blocks=[]

    for fewshot in fewshots:
        blocks.append(
            "### Example Output\n"
            +json.dumps(fewshot,ensure_ascii=False)
            +"\n"
        )
    return "\n".join(blocks)

