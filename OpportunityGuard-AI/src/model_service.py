"""
model_service.py — Load the trained OpportunityGuard pipeline and run inference.

The fitted scikit-learn Pipeline (TF-IDF + encoders + LogisticRegression) is
loaded once at startup and cached.  No preprocessing is refit at inference time.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Artifact paths ────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = _ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "opportunityguard_model.joblib"
CONFIG_PATH = ARTIFACTS_DIR / "opportunityguard_config.json"


# ─── Config loading ────────────────────────────────────────────────────────────
def load_config() -> dict[str, Any]:
    """Return the exported config dictionary."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with CONFIG_PATH.open() as fh:
        return json.load(fh)


# ─── Model loading (cached externally via st.cache_resource in app) ────────────
def load_model() -> Any:
    """Load and return the fitted scikit-learn pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}\n"
            "Please copy opportunityguard_model.joblib into the artifacts/ directory."
        )
    try:
        pipeline = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
        return pipeline
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {MODEL_PATH}: {exc}\n"
            "This may be a version mismatch. Ensure scikit-learn and numpy "
            "versions match those used during training."
        ) from exc


# ─── Feature assembly ──────────────────────────────────────────────────────────
def build_model_row(listing: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """
    Build a single-row DataFrame whose columns exactly match config['model_fields'].

    The pipeline was trained on 'model_text' (concatenated text fields) plus
    categorical and numeric columns.  All missing optional fields are filled
    with empty string or 0 as appropriate.
    """
    text_fields: list[str] = config.get("text_fields", [])
    model_fields: list[str] = config.get("model_fields", [])
    categorical_fields: list[str] = config.get("categorical_fields", [])
    numeric_fields: list[str] = config.get("numeric_fields", [])

    # Concatenate text fields → model_text
    text_parts = []
    for field in text_fields:
        val = listing.get(field, "") or ""
        text_parts.append(str(val).strip())
    model_text = " ".join(filter(None, text_parts))

    row: dict[str, Any] = {"model_text": model_text}

    for field in categorical_fields:
        row[field] = str(listing.get(field, "") or "")

    for field in numeric_fields:
        raw = listing.get(field, 0)
        try:
            row[field] = int(raw)
        except (TypeError, ValueError):
            row[field] = 0

    # Return only the columns the pipeline expects, in the correct order
    ordered = {col: row.get(col, "") for col in model_fields}
    return pd.DataFrame([ordered])


# ─── Inference ─────────────────────────────────────────────────────────────────
def predict(
    listing: dict[str, Any],
    pipeline: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the pipeline on a single listing.

    Returns a dict with:
      model_risk_score  — raw probability (0-1) from predict_proba
      model_risk_percent — 0-100 rounded to 2 dp
      model_flagged     — True if probability >= decision_threshold
      model_threshold   — the configured threshold
      model_version     — from config
    """
    threshold: float = config.get("decision_threshold", 0.5)
    version: str = config.get("model_version", "unknown")

    df = build_model_row(listing, config)

    try:
        proba = pipeline.predict_proba(df)[0, 1]
    except AttributeError:
        # Fallback for SVM-like estimators without predict_proba
        decision = pipeline.decision_function(df)[0]
        # Map decision function to [0, 1] via sigmoid
        proba = float(1 / (1 + np.exp(-decision)))

    model_risk_score = float(proba)
    model_risk_percent = round(model_risk_score * 100, 2)
    model_flagged = model_risk_score >= threshold

    return {
        "model_risk_score": model_risk_score,
        "model_risk_percent": model_risk_percent,
        "model_flagged": model_flagged,
        "model_threshold": threshold,
        "model_version": version,
        "model_prediction": "Flagged by model" if model_flagged else "Not flagged by model",
    }
