"""
risk_scoring.py — Combine ML model output and rule-based score into a
composite risk indicator.

Formula (from config):
    composite_score = round(0.65 * model_risk_percent + 0.35 * rule_score)

Thresholds (configurable):
    composite < 30  → Lower detected risk
    30 ≤ composite < 60  → Review recommended
    composite ≥ 60  → High risk — verify before proceeding
"""
from __future__ import annotations

from typing import Any


# ─── Configurable thresholds ────────────────────────────────────────────────────
LOWER_RISK_THRESHOLD = 30
HIGH_RISK_THRESHOLD = 60
RULE_REVIEW_THRESHOLD = 35

CATEGORY_LOWER = "Lower detected risk — still verify independently"
CATEGORY_REVIEW = "Review recommended — investigate before proceeding"
CATEGORY_HIGH = "High risk — verify before proceeding"


def compute_composite_score(
    model_risk_percent: float,
    rule_score: float,
    ml_weight: float = 0.65,
    rule_weight: float = 0.35,
) -> int:
    """Return the composite risk score (0-100 integer)."""
    raw = ml_weight * model_risk_percent + rule_weight * rule_score
    return int(round(min(100, max(0, raw))))


def classify_composite(composite_score: int) -> str:
    """Map composite score to a human-readable category."""
    if composite_score < LOWER_RISK_THRESHOLD:
        return CATEGORY_LOWER
    elif composite_score < HIGH_RISK_THRESHOLD:
        return CATEGORY_REVIEW
    else:
        return CATEGORY_HIGH


def needs_human_review(
    model_flagged: bool,
    rule_score: float,
    composite_score: int,
) -> bool:
    """Return True when a listing should be queued for human review."""
    return model_flagged or rule_score >= RULE_REVIEW_THRESHOLD or composite_score >= LOWER_RISK_THRESHOLD


def generate_risk_explanation(
    model_risk_percent: float,
    rule_score: float,
    composite_score: int,
    model_flagged: bool,
    model_threshold: float = 0.78,
    ml_weight: float = 0.65,
    rule_weight: float = 0.35,
    lower_threshold: int = LOWER_RISK_THRESHOLD,
    high_threshold: int = HIGH_RISK_THRESHOLD,
    rule_review_threshold: float = RULE_REVIEW_THRESHOLD,
) -> str:
    """
    Generate a dynamic, mathematically accurate explanation of the risk assessment.

    Explains:
      1. Formula breakdown: round(ml_weight * model_risk_percent + rule_weight * rule_score)
      2. ML decision vs. composite risk category as separate decisions.
      3. Precise human-review routing conditions.
      4. Cautious risk-indicator language (not proof of fraud).
    """
    ml_contrib = ml_weight * model_risk_percent
    rule_contrib = rule_weight * rule_score
    raw_sum = ml_contrib + rule_contrib
    thresh_pct = model_threshold * 100.0

    if composite_score < lower_threshold:
        band_name = f"Lower detected risk (0–{lower_threshold - 1})"
        band_guidance = "No major heuristic warnings or significant ML anomaly patterns detected. Standard verification still recommended."
    elif composite_score < high_threshold:
        band_name = f"Review recommended ({lower_threshold}–{high_threshold - 1})"
        band_guidance = "Moderate risk indicators detected. Manual review and employer verification recommended before proceeding."
    else:
        band_name = f"High risk ({high_threshold}–100)"
        band_guidance = "Elevated risk indicators detected. Thorough independent verification required before proceeding."

    routing_triggers: list[str] = []
    if model_flagged:
        routing_triggers.append(f"ML model flagged listing (score {model_risk_percent:.1f}% ≥ {thresh_pct:.1f}%)")
    if rule_score >= rule_review_threshold:
        routing_triggers.append(f"Rule score {rule_score:.0f} reached rule threshold ({rule_review_threshold:.0f} pts)")
    if composite_score >= lower_threshold:
        routing_triggers.append(f"Composite score {composite_score} reached review threshold ({lower_threshold} pts)")

    review_needed = len(routing_triggers) > 0

    explanation = (
        f"### 📐 Dynamic Risk Scoring & Decision Breakdown\n\n"
        f"**1. Composite Formula & Calculation**\n"
        f"- **Configured Formula:** `composite = round({ml_weight:.2f} × model_risk_percent + {rule_weight:.2f} × rule_score)`\n"
        f"- **Calculation for this listing:**\n"
        f"  - ML model contribution: `{ml_weight:.2f} × {model_risk_percent:.1f}% = {ml_contrib:.2f}`\n"
        f"  - Rule engine contribution: `{rule_weight:.2f} × {rule_score:.0f} = {rule_contrib:.2f}`\n"
        f"  - Raw combined score: `{ml_contrib:.2f} + {rule_contrib:.2f} = {raw_sum:.2f}`\n"
        f"  - Clamped & rounded composite score: **`{composite_score}` / 100**\n"
        f"  - Assigned Risk Band: **{band_name}** — *{band_guidance}*\n\n"
        f"**2. ML Flag vs. Composite Risk Category (Separate Decisions)**\n"
        f"- **ML Model Decision:** The machine learning model uses a statistical decision threshold of **{model_threshold:.2f}** ({thresh_pct:.1f}%), "
        f"tuned on historical training listings to identify vocabulary and metadata anomalies. Current ML flag status: **{'Flagged' if model_flagged else 'Not Flagged'}** ({model_risk_percent:.1f}%).\n"
        f"- **Composite Risk Category:** Evaluates multi-tier risk combining both the ML model (65% weight) and deterministic heuristic rules (35% weight):\n"
        f"  - `0 – 29`: Lower detected risk\n"
        f"  - `30 – 59`: Review recommended\n"
        f"  - `60 – 100`: High risk\n"
        f"- **Why these are separate decisions:**\n"
        f"  - The ML flag is a binary decision indicating lexical/categorical similarity to historical training data.\n"
        f"  - The composite score combines ML signals with deterministic heuristic rules.\n"
        f"  - For example, an unflagged ML score of 45% (< 78%) combined with a rule score of 70 yields `round(0.65 × 45 + 0.35 × 70) = round(29.25 + 24.50) = 54`, "
        f"which falls in the **Review recommended** band (30–59), NOT the High Risk band.\n"
        f"  - To reach the **High Risk** band (≥ 60) with an ML score of 45%, the rule score must be at least 88 (since `round(0.65 × 45 + 0.35 × 88) = round(29.25 + 30.80) = 60`).\n"
        f"  - Conversely, an ML-flagged listing (e.g. 80% ≥ 78%) with 0 rule warnings yields `round(0.65 × 80 + 0) = 52`, placing it in the **Review recommended** composite band.\n\n"
        f"**3. Human-Review Routing Conditions**\n"
        f"A listing is routed to the human-review workflow when **ANY** of the following criteria is met:\n"
        f"1. `model_flagged == True` (ML risk score ≥ {model_threshold:.2f} / {thresh_pct:.1f}%) — *(Status: {model_flagged})*\n"
        f"2. `rule_score ≥ {rule_review_threshold:.0f}` (significant heuristic warnings triggered) — *(Status: {rule_score:.0f} ≥ {rule_review_threshold:.0f} → {rule_score >= rule_review_threshold})*\n"
        f"3. `composite_score ≥ {lower_threshold}` (composite category is Review Recommended or High Risk) — *(Status: {composite_score} ≥ {lower_threshold} → {composite_score >= lower_threshold})*\n\n"
        f"- **Routing Decision for this listing:** **{'Review Recommended — Queued for Human Review' if review_needed else 'Human Review Not Required'}**\n"
        + (f"- **Active triggers:** {'; '.join(routing_triggers)}\n\n" if routing_triggers else "- **Active triggers:** None\n\n")
        + f"⚠️ *Note: OpportunityGuard AI provides assistive risk indicators and does not prove fraud or guarantee legitimacy. Always verify employers through independent official channels.*"
    )
    return explanation


def build_full_assessment(
    listing: dict[str, Any],
    model_result: dict[str, Any],
    rule_result: dict[str, Any],
    similar_listings: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble the complete assessment response dict mirroring sample_response.json.
    """
    model_risk_percent: float = model_result.get("model_risk_percent", 0.0)
    rule_score: float = rule_result.get("rule_score", 0)
    model_flagged: bool = model_result.get("model_flagged", False)
    model_threshold: float = model_result.get("model_threshold", 0.78)

    composite = compute_composite_score(model_risk_percent, rule_score)
    category = classify_composite(composite)
    review_needed = needs_human_review(
        model_flagged,
        rule_score,
        composite,
    )

    explanation = generate_risk_explanation(
        model_risk_percent=model_risk_percent,
        rule_score=rule_score,
        composite_score=composite,
        model_flagged=model_flagged,
        model_threshold=model_threshold,
    )

    return {
        "model_version": model_result.get("model_version", "unknown"),
        "model_risk_score": model_result.get("model_risk_score", 0.0),
        "model_risk_percent": model_risk_percent,
        "model_prediction": model_result.get("model_prediction", ""),
        "model_flagged": model_flagged,
        "model_threshold": model_threshold,
        "rule_score": rule_score,
        "composite_risk_score": composite,
        "assessment": category,
        "needs_human_review": review_needed,
        "warnings": rule_result.get("warnings", []),
        "similar_listings": similar_listings,
        "formula": "composite = round(0.65 × model_risk_percent + 0.35 × rule_score)",
        "explanation": explanation,
        "disclaimer": (
            "Risk indicator only, not proof of fraud. "
            "Verify independently before making any decision."
        ),
    }
