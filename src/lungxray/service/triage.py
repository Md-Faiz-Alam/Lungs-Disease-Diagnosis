"""Worklist triage: which studies a clinician should look at first.

urgent  - model predicts disease (COVID-19 / Pneumonia) with confidence above the referral threshold
review  - model is unsure (calibrated confidence below threshold) or the input raised warnings
routine - model confidently predicts Normal
"""

from __future__ import annotations

DISEASE_LABELS = {"COVID19", "PNEUMONIA"}
TRIAGE_RANK = {"urgent": 0, "review": 1, "routine": 2}


def triage(prediction: str, needs_review: bool, warnings: list[str]) -> str:
    if needs_review:
        return "review"
    if prediction in DISEASE_LABELS:
        return "urgent"  # a confident disease call is prioritised even if the input raised warnings
    return "review" if warnings else "routine"
