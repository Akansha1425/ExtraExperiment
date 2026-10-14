"""
duplicate_detector.py — Detect similar descriptions using the exported
TF-IDF + cosine-similarity index.

The joblib artifact contains a dict:
  {
    "tfidf": fitted TfidfVectorizer,
    "matrix": sparse matrix (n_docs × n_features),
    "descriptions": list[str]   (raw preview text)
  }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = _ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "opportunityguard_duplicate_index.joblib"


def load_duplicate_index() -> dict[str, Any]:
    """Load the pre-built duplicate-detection index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Duplicate index not found: {INDEX_PATH}\n"
            "Please copy opportunityguard_duplicate_index.joblib into the artifacts/ directory."
        )
    try:
        index = joblib.load(INDEX_PATH)
        logger.info("Duplicate index loaded: %d reference descriptions", len(index.get("descriptions", [])))
        return index
    except Exception as exc:
        raise RuntimeError(f"Failed to load duplicate index: {exc}") from exc


def find_similar(
    description: str,
    index: dict[str, Any],
    top_k: int = 5,
    threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Find the top_k most similar indexed descriptions.

    Returns a list of dicts:
      {similarity, reference_index, description_preview}

    Only results with similarity >= threshold are returned.

    The index may use either:
      - keys: 'tfidf', 'matrix', 'descriptions'   (expected schema)
      - keys: 'vectorizer', 'matrix', 'reference_texts', 'reference_indices'  (actual artifact)
    Both are handled transparently.
    """
    if not description or not description.strip():
        return []

    # Support both key naming conventions
    tfidf = index.get("tfidf") or index.get("vectorizer")
    matrix = index.get("matrix")
    descriptions: list[str] = index.get("descriptions") or index.get("reference_texts", [])
    ref_indices: list[int] | None = index.get("reference_indices")

    if tfidf is None or matrix is None or len(descriptions) == 0:
        logger.warning("Duplicate index is incomplete or empty.")
        return []

    try:
        query_vec = tfidf.transform([description])
    except Exception as exc:
        logger.warning("Failed to transform query text: %s", exc)
        return []

    try:
        # Sparse matrix dot product for cosine similarity (matrix is already L2-normalised)
        scores = (matrix @ query_vec.T).toarray().flatten()
    except Exception as exc:
        logger.warning("Similarity computation failed: %s", exc)
        return []

    # Select top_k results above threshold
    indices = np.argsort(scores)[::-1][:top_k * 3]  # extra buffer before threshold filter
    results = []
    for idx in indices:
        sim = float(scores[idx])
        if sim < threshold:
            continue
        preview = descriptions[idx] if idx < len(descriptions) else ""
        preview_text = str(preview)[:300].strip()
        # Use original dataset index if the artifact provides a mapping
        actual_idx = int(ref_indices[idx]) if (ref_indices is not None and idx < len(ref_indices)) else int(idx)
        results.append(
            {
                "similarity": round(sim, 4),
                "reference_index": actual_idx,
                "description_preview": preview_text,
            }
        )
        if len(results) >= top_k:
            break

    return results
