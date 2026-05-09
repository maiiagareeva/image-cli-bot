from __future__ import annotations

import re
from typing import Any, Dict, List


SECTION_HEADERS = [
    "disease",
    "indicators",
    "recommended checks",
    "evidence",
]


def normalize_text(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{2,}", "\n", x)
    return x.strip()


def normalize_inline_text(x):
    return " ".join(normalize_text(x).lower().split())


def normalize_disease(x):
    s = normalize_inline_text(x)
    s = s.replace("-", " ").replace("_", " ")

    if not s:
        return "unknown"
    if "healthy" in s:
        return "healthy"
    if "downy" in s:
        return "downy mildew"
    if "powdery" in s:
        return "powdery mildew"
    return "unknown"


def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        k = normalize_inline_text(x)
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def find_section_positions(text):
    positions = {}
    for header in SECTION_HEADERS:
        pattern = rf"(^|\n)\s*{re.escape(header)}\s*:"
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            positions[header] = m.start()
    return positions


def slice_section(text, section_name):
    if not text:
        return ""

    positions = find_section_positions(text)
    if section_name not in positions:
        return ""

    start = positions[section_name]
    later = sorted(
        [(name, pos) for name, pos in positions.items() if pos > start],
        key=lambda x: x[1],
    )
    end = later[0][1] if later else len(text)

    chunk = text[start:end].strip()
    chunk = re.sub(
        rf"^\s*{re.escape(section_name)}\s*:\s*",
        "",
        chunk,
        flags=re.IGNORECASE,
    ).strip()
    return chunk


def has_section(text, section_name):
    if not text:
        return False
    pattern = rf"(^|\n)\s*{re.escape(section_name)}\s*:"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_disease(text):
    chunk = slice_section(text, "disease")
    if chunk:
        first_line = chunk.split("\n")[0].strip()
        first_line = re.split(r"[.;,]", first_line)[0].strip()
        return normalize_disease(first_line)
    return normalize_disease(text)


def clean_bullet_line(line):
    line = line.strip()
    line = re.sub(r"^[-*•\d\.\)\(]+\s*", "", line)
    return normalize_inline_text(line)


def extract_list_section(text, section_name):
    chunk = slice_section(text, section_name)
    if not chunk:
        return []

    lines = [ln.strip() for ln in chunk.split("\n") if ln.strip()]
    items: List[str] = []

    bullet_count = 0
    for ln in lines:
        if re.match(r"^\s*[-*•\d\.\)\(]+\s+", ln):
            bullet_count += 1

    if bullet_count >= 1:
        for ln in lines:
            cleaned = clean_bullet_line(ln)
            if cleaned:
                items.append(cleaned)
        return dedup_preserve_order(items)

    raw = chunk.replace("\n", " ")
    parts = re.split(r";|(?<=[.!?])\s+", raw)
    for p in parts:
        cleaned = clean_bullet_line(p)
        if cleaned:
            items.append(cleaned)

    return dedup_preserve_order(items)


def extract_evidence(text):
    chunk = slice_section(text, "evidence")
    if chunk:
        return normalize_inline_text(chunk)
    return ""


def parse_plain_answer(text):
    text = normalize_text(text)
    return {
        "disease": extract_disease(text),
        "indicators": extract_list_section(text, "indicators"),
        "recommended_checks": extract_list_section(text, "recommended checks"),
        "evidence": extract_evidence(text),
        "has_disease_section": has_section(text, "disease"),
        "has_indicators_section": has_section(text, "indicators"),
        "has_checks_section": has_section(text, "recommended checks"),
        "has_evidence_section": has_section(text, "evidence"),
    }
