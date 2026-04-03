from __future__ import annotations
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")
bertscore = evaluate.load("bertscore")

REQUIRED_KEYS = ["disease", "symptoms", "recommended_checks", "evidence"]

def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(x.strip().lower().split())


def normalize_disease(x: Any) -> str:
    s = normalize_text(x)
    s = s.replace("-", " ").replace("_", " ")
    if "healthy" in s:
        return "healthy"
    if "downy" in s:
        return "downy mildew"
    if "powdery" in s:
        return "powdery mildew"
    if not s:
        return "unknown"
    return "unknown"


def normalize_list_field(x: Any) -> List[str]:
    if x is None:
        return []

    if isinstance(x, str):
        items = [x]
    elif isinstance(x, list):
        items = x
    else:
        items = [str(x)]

    normed = []
    for item in items:
        s = normalize_text(item)
        if s:
            normed.append(s)

    seen = set()
    uniq = []
    for s in normed:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def schema_valid(obj: Optional[Dict[str, Any]]) -> bool:
    if obj is None or not isinstance(obj, dict):
        return False

    for k in REQUIRED_KEYS:
        if k not in obj:
            return False

    if not isinstance(obj["disease"], str):
        return False
    if not isinstance(obj["symptoms"], list):
        return False
    if not isinstance(obj["recommended_checks"], list):
        return False
    if not isinstance(obj["evidence"], str):
        return False

    return True


def full_json_exact_match(pred_obj: Optional[Dict[str, Any]], ref_obj: Optional[Dict[str, Any]]) -> bool:
    if pred_obj is None or ref_obj is None:
        return False

    pred_norm = {
        "disease": normalize_disease(pred_obj.get("disease")),
        "symptoms": normalize_list_field(pred_obj.get("symptoms")),
        "recommended_checks": normalize_list_field(pred_obj.get("recommended_checks")),
        "evidence": normalize_text(pred_obj.get("evidence")),
    }
    ref_norm = {
        "disease": normalize_disease(ref_obj.get("disease")),
        "symptoms": normalize_list_field(ref_obj.get("symptoms")),
        "recommended_checks": normalize_list_field(ref_obj.get("recommended_checks")),
        "evidence": normalize_text(ref_obj.get("evidence")),
    }
    return pred_norm == ref_norm


def set_prf(pred_items: List[str], ref_items: List[str]) -> Tuple[float, float, float]:
    pred_set = set(pred_items)
    ref_set = set(ref_items)

    if len(pred_set) == 0 and len(ref_set) == 0:
        return 1.0, 1.0, 1.0
    if len(pred_set) == 0 and len(ref_set) > 0:
        return 0.0, 0.0, 0.0
    if len(pred_set) > 0 and len(ref_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & ref_set)
    prec = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    rec = tp / len(ref_set) if len(ref_set) > 0 else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def evaluate_predictions(records: List[Dict[str, Any]], bertscore_lang: str = "en") -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        raise ValueError("No records found in predictions file.")

    parse_ok_count = 0
    schema_ok_count = 0
    full_exact_count = 0

    y_true = []
    y_pred = []

    symptoms_prec = []
    symptoms_rec = []
    symptoms_f1 = []

    checks_prec = []
    checks_rec = []
    checks_f1 = []

    evidence_pred = []
    evidence_ref = []

    whole_pred_text = []
    whole_ref_text = []

    for rec in records:
        ref_text = rec.get("reference_text", "")
        pred_text = rec.get("prediction_text", "")

        ref_obj = rec.get("reference_json")
        pred_obj = rec.get("prediction_json")

        # if not pre-parsed, parse on the fly
        if ref_obj is None:
            ref_obj = safe_parse_json(ref_text)
        if pred_obj is None:
            pred_obj = safe_parse_json(pred_text)

        if pred_obj is not None:
            parse_ok_count += 1
        if schema_valid(pred_obj):
            schema_ok_count += 1
        if full_json_exact_match(pred_obj, ref_obj):
            full_exact_count += 1

        # disease metrics
        ref_d = normalize_disease(ref_obj.get("disease") if ref_obj else None)
        pred_d = normalize_disease(pred_obj.get("disease") if pred_obj else None)
        y_true.append(ref_d)
        y_pred.append(pred_d)

        # symptoms set metrics
        ref_sym = normalize_list_field(ref_obj.get("symptoms") if ref_obj else None)
        pred_sym = normalize_list_field(pred_obj.get("symptoms") if pred_obj else None)
        p, r, f = set_prf(pred_sym, ref_sym)
        symptoms_prec.append(p)
        symptoms_rec.append(r)
        symptoms_f1.append(f)

        ref_chk = normalize_list_field(ref_obj.get("recommended_checks") if ref_obj else None)
        pred_chk = normalize_list_field(pred_obj.get("recommended_checks") if pred_obj else None)
        p, r, f = set_prf(pred_chk, ref_chk)
        checks_prec.append(p)
        checks_rec.append(r)
        checks_f1.append(f)

        evidence_ref.append(normalize_text(ref_obj.get("evidence") if ref_obj else ""))
        evidence_pred.append(normalize_text(pred_obj.get("evidence") if pred_obj else ""))

        whole_ref_text.append(ref_text if isinstance(ref_text, str) else json.dumps(ref_text, ensure_ascii=False))
        whole_pred_text.append(pred_text if isinstance(pred_text, str) else json.dumps(pred_text, ensure_ascii=False))

    disease_acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm_labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels).tolist()

    evidence_rouge = rouge.compute(
        predictions=evidence_pred,
        references=evidence_ref,
        use_stemmer=True,
    )
    evidence_bertscore = bertscore.compute(
        predictions=evidence_pred,
        references=evidence_ref,
        lang=bertscore_lang,
    )
    evidence_bertscore_f1 = mean([float(x) for x in evidence_bertscore["f1"]])

    whole_bleu = bleu.compute(
        predictions=whole_pred_text,
        references=[[r] for r in whole_ref_text],
    )
    whole_rouge = rouge.compute(
        predictions=whole_pred_text,
        references=whole_ref_text,
        use_stemmer=True,
    )

    results = {
        "num_samples": n,
        "valid_json_rate": parse_ok_count / n,
        "schema_valid_rate": schema_ok_count / n,
        "full_json_exact_match_rate": full_exact_count / n,

        "disease_accuracy": float(disease_acc),
        "disease_macro_precision": float(p_macro),
        "disease_macro_recall": float(r_macro),
        "disease_macro_f1": float(f1_macro),

        "symptoms_precision": mean(symptoms_prec),
        "symptoms_recall": mean(symptoms_rec),
        "symptoms_f1": mean(symptoms_f1),

        "checks_precision": mean(checks_prec),
        "checks_recall": mean(checks_rec),
        "checks_f1": mean(checks_f1),

        "evidence_rouge1": float(evidence_rouge["rouge1"]),
        "evidence_rouge2": float(evidence_rouge["rouge2"]),
        "evidence_rougeL": float(evidence_rouge["rougeL"]),
        "evidence_bertscore_f1": float(evidence_bertscore_f1),

        "whole_bleu": float(whole_bleu["score"]),
        "whole_rouge1": float(whole_rouge["rouge1"]),
        "whole_rouge2": float(whole_rouge["rouge2"]),
        "whole_rougeL": float(whole_rouge["rougeL"]),

        "confusion_matrix_labels": cm_labels,
        "confusion_matrix": cm,
        "per_class_report": per_class,
    }
    return results


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to save evaluation json")
    parser.add_argument("--bertscore_lang", type=str, default="en")
    args = parser.parse_args()

    records = load_jsonl(args.predictions)
    results = evaluate_predictions(records, bertscore_lang=args.bertscore_lang)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()