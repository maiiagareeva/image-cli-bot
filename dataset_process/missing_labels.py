import os
import csv
import json
import base64
from pathlib import Path

DATA_ROOT = Path("GreenHouse_img")
MISSED_DIR = [DATA_ROOT / "missed"]
LABELS_DIR = [DATA_ROOT / "labels"]
IMAGE_DIRS = [DATA_ROOT]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def iter_all_labels():
    for d in LABELS_DIR:
        if not d.exists():
                continue
        split_name = d.name
        print()
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in '.txt':
                yield p.stem


def iter_all_missed():
    for d in MISSED_DIR:
        if not d.exists():
                continue
        split_name = d.name
        print()
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in '.json':
                yield p.stem


def iter_all_images():
    for d in IMAGE_DIRS:
        if not d.exists():
            continue
        split_name = d.name
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p.stem

# labels = list(iter_all_labels())
# print()
# images = list(iter_all_images())

# not_included = []
# flag = True    
# for i in range(len(labels)):
#      b = labels[i]
#      if '(' in b:
#         print(b)
#      if b not in images:
#         not_included.append(b)
#         print(f'lost object: {b}')
#         flag = False
# if flag:
#     print('all included')

all_missed_proccessed = list(iter_all_missed())
print(len(all_missed_proccessed))
print()
with open('missing_png_json_report.csv', 'r') as file:
    lis = file.readlines()[1:]
    missing_labels = [line.split(',')[0] for line in lis]
print()

# not_included = []
# flag = True    
# for i in range(len(missing_labels)):
#      b = missing_labels[i]
#      if b not in all_missed_proccessed:
#         not_included.append(b)
#         print(f'lost object: {b}')
#         flag = False
# if flag:
#     print('all included')

