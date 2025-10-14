import pandas as pd
import argparse
from pathlib import Path
import re
import unicodedata
import nltk


nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
DOMAIN_STOPWORDS = set()

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


df = pd.read_csv(args.manifest_out)
dic = {}
for row in df["texts"]:
    s = unicodedata.normalize("NFC", str(row))
    s = s.replace("\u00A0", " ").strip().lower()  # non-breaking spaces turn into space
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    curr = row.split()
    all_words = [w for w in s.split() if w not in STOPWORDS] # creating a new filtered text from the old one
    for ch in all_words:
        dic[ch] = dic.get(ch, 0) + 1
    sorted_dict = dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))
    dic.update(sorted_dict)

maxi = 15
DOMAIN_STOPWORDS |= {k for k, v in dic.items() if v > maxi}
print(DOMAIN_STOPWORDS)
# print(sorted(list(dic.values())))