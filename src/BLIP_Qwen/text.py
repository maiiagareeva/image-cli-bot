from __future__ import annotations
import json

def answer_json_to_stage1_text(answer_text: str) -> str:
    try:
        obj = json.loads(answer_text)
    except Exception:
        return answer_text

    disease = obj.get("disease", "")
    symptoms = obj.get("symptoms", [])
    checks = obj.get("recommended_checks", [])
    evidence = obj.get("evidence", "")

    if isinstance(symptoms, list):
        symptoms = ", ".join(str(x) for x in symptoms)
    if isinstance(checks, list):
        checks = ", ".join(str(x) for x in checks)

    text = (
        f"Disease: {disease}. "
        f"Symptoms: {symptoms}. "
        f"Recommended checks: {checks}. "
        f"Evidence: {evidence}."
    )
    return " ".join(text.split())