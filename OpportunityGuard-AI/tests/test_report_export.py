"""
tests/test_report_export.py — Tests for JSON/CSV export.
"""
import sys, json, csv, io
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.report_export import to_json, to_csv

SAMPLE_LISTING = {"title": "Dev Intern", "company_name": "Acme"}
SAMPLE_ASSESSMENT = {
    "model_version": "test-v1",
    "model_risk_percent": 20.0,
    "model_prediction": "Not flagged",
    "model_threshold": 0.78,
    "rule_score": 10,
    "composite_risk_score": 17,
    "assessment": "Lower detected risk",
    "needs_human_review": False,
    "formula": "round(0.65*m + 0.35*r)",
    "disclaimer": "Risk indicator only.",
    "warnings": [{"code": "test", "severity": "low", "points": 10,
                   "field": "description", "evidence": "...", "recommendation": "..."}],
    "similar_listings": [{"similarity": 0.5, "reference_index": 1, "description_preview": "preview"}],
}


class TestToJson:
    def test_returns_string(self):
        result = to_json(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        assert isinstance(result, str)

    def test_valid_json(self):
        result = to_json(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_contains_listing_data(self):
        result = to_json(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        parsed = json.loads(result)
        assert parsed["listing"]["title"] == "Dev Intern"

    def test_contains_assessment_data(self):
        result = to_json(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        parsed = json.loads(result)
        assert parsed["assessment"]["composite_risk_score"] == 17

    def test_contains_timestamp(self):
        result = to_json(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        parsed = json.loads(result)
        assert "exported_at" in parsed


class TestToCsv:
    def test_returns_string(self):
        result = to_csv(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        assert isinstance(result, str)

    def test_parseable_as_csv(self):
        result = to_csv(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        rows = list(csv.reader(io.StringIO(result)))
        assert len(rows) > 5

    def test_contains_composite_score(self):
        result = to_csv(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        assert "17" in result

    def test_contains_warning_code(self):
        result = to_csv(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        assert "test" in result

    def test_similar_listing_preview_included(self):
        result = to_csv(SAMPLE_ASSESSMENT, SAMPLE_LISTING)
        assert "preview" in result
