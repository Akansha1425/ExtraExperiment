"""
report_export.py — Generate downloadable assessment reports in JSON or CSV.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any


def to_json(assessment: dict[str, Any], listing: dict[str, Any]) -> str:
    """Return a JSON string of the full assessment report."""
    report = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "listing": listing,
        "assessment": assessment,
    }
    return json.dumps(report, indent=2, default=str)


def to_csv(assessment: dict[str, Any], listing: dict[str, Any]) -> str:
    """Return a CSV string summarising key fields from the assessment."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Field", "Value"])
    writer.writerow(["Exported at", datetime.now(timezone.utc).isoformat()])
    writer.writerow([])

    writer.writerow(["=== LISTING ==="])
    for key, val in listing.items():
        writer.writerow([key, str(val)])
    writer.writerow([])

    writer.writerow(["=== ASSESSMENT ==="])
    writer.writerow(["model_version", assessment.get("model_version", "")])
    writer.writerow(["model_risk_percent", assessment.get("model_risk_percent", "")])
    writer.writerow(["model_prediction", assessment.get("model_prediction", "")])
    writer.writerow(["model_threshold", assessment.get("model_threshold", "")])
    writer.writerow(["rule_score", assessment.get("rule_score", "")])
    writer.writerow(["composite_risk_score", assessment.get("composite_risk_score", "")])
    writer.writerow(["assessment_category", assessment.get("assessment", "")])
    writer.writerow(["needs_human_review", assessment.get("needs_human_review", "")])
    writer.writerow(["formula", assessment.get("formula", "")])
    writer.writerow(["disclaimer", assessment.get("disclaimer", "")])
    writer.writerow([])

    writer.writerow(["=== WARNINGS ==="])
    writer.writerow(["Code", "Severity", "Points", "Field", "Evidence", "Recommendation"])
    for w in assessment.get("warnings", []):
        writer.writerow([
            w.get("code", ""),
            w.get("severity", ""),
            w.get("points", ""),
            w.get("field", ""),
            w.get("evidence", ""),
            w.get("recommendation", ""),
        ])
    writer.writerow([])

    writer.writerow(["=== SIMILAR LISTINGS ==="])
    writer.writerow(["Rank", "Similarity", "Reference Index", "Description Preview"])
    for i, s in enumerate(assessment.get("similar_listings", []), 1):
        writer.writerow([
            i,
            s.get("similarity", ""),
            s.get("reference_index", ""),
            s.get("description_preview", "")[:200],
        ])

    return buf.getvalue()
