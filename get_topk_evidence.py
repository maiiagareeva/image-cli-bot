from metadata_manifest import store as metadata_csv
import csv
from clip_zero_shot import image_text_topk

def load_all_texts_from_manifest(csv_path):
    all_text=[]
    with open(csv_path,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for text in row["texts"].split('|'):
                text = " ".join(text.strip().split())
                if text:
                    all_text.append(text)
    return all_text

ALL_TEXT=load_all_texts_from_manifest(metadata_csv)
def clip_topk_evidence(image_path:str,k:int):
    topk=image_text_topk([image_path],ALL_TEXT,k)
    # [[("text1", 1.1), ("text2", 2.2), ("text3", 3.3)]]
    topk_= topk[0] if topk else []
    results=["topk evidence from CLIP:"]
    for text,score in topk_:
        results.append(f"- {text} (clip={score:.3f})")
    return "\n".join(results)
