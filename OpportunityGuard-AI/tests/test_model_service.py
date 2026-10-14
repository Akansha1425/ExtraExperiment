"""
tests/test_model_service.py — Tests for model loading and inference.
"""
import sys
from pathlib import Path
import pytest
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.model_service import load_model, load_config, build_model_row, predict


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def pipeline():
    return load_model()


class TestLoadConfig:
    def test_config_loads(self, config):
        assert isinstance(config, dict)

    def test_config_has_required_keys(self, config):
        for key in ("model_fields", "text_fields", "categorical_fields", "numeric_fields",
                    "decision_threshold", "composite_score_formula"):
            assert key in config, f"Config missing key: {key}"

    def test_decision_threshold_range(self, config):
        t = config["decision_threshold"]
        assert 0.0 < t < 1.0


class TestLoadModel:
    def test_model_loads(self, pipeline):
        assert pipeline is not None

    def test_model_has_predict_proba(self, pipeline):
        assert hasattr(pipeline, "predict_proba"), "Model must support predict_proba"


class TestBuildModelRow:
    def test_full_listing_builds_correct_columns(self, config):
        listing = {
            "title": "Intern", "description": "Build software", "company_profile": "Tech company",
            "requirements": "Python", "benefits": "Mentorship", "salary_range": "",
            "employment_type": "Internship", "required_experience": "Entry level",
            "required_education": "Bachelor's Degree", "industry": "IT",
            "function": "Engineering", "location": "Remote",
            "telecommuting": 1, "has_company_logo": 1, "has_questions": 0,
        }
        df = build_model_row(listing, config)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == config["model_fields"]
        assert len(df) == 1

    def test_empty_listing_fills_defaults(self, config):
        df = build_model_row({}, config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_model_text_concatenates_text_fields(self, config):
        listing = {"title": "Engineer", "description": "Build things", "company_profile": "Startup"}
        df = build_model_row(listing, config)
        text = df["model_text"].iloc[0]
        assert "Engineer" in text
        assert "Build things" in text


class TestPredict:
    SAMPLE_LISTING = {
        "title": "Remote Software Development Intern",
        "company_profile": "Small software services company.",
        "description": "Build web applications with a mentor.",
        "requirements": "Basic JavaScript and HTML.",
        "benefits": "Mentorship",
        "salary_range": "",
        "employment_type": "Internship",
        "required_experience": "Entry level",
        "required_education": "Bachelor's Degree",
        "industry": "IT",
        "function": "Engineering",
        "location": "Remote",
        "telecommuting": 1,
        "has_company_logo": 1,
        "has_questions": 0,
    }

    def test_predict_returns_dict(self, pipeline, config):
        result = predict(self.SAMPLE_LISTING, pipeline, config)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, pipeline, config):
        result = predict(self.SAMPLE_LISTING, pipeline, config)
        for key in ("model_risk_score", "model_risk_percent", "model_flagged",
                    "model_threshold", "model_version", "model_prediction"):
            assert key in result

    def test_risk_score_in_range(self, pipeline, config):
        result = predict(self.SAMPLE_LISTING, pipeline, config)
        assert 0.0 <= result["model_risk_score"] <= 1.0
        assert 0.0 <= result["model_risk_percent"] <= 100.0

    def test_predict_empty_listing_does_not_crash(self, pipeline, config):
        result = predict({}, pipeline, config)
        assert 0.0 <= result["model_risk_score"] <= 1.0

    def test_predict_partial_listing(self, pipeline, config):
        listing = {"description": "Work from home, guaranteed income, no experience needed"}
        result = predict(listing, pipeline, config)
        assert isinstance(result["model_risk_score"], float)

    def test_missing_artifact_raises(self, tmp_path):
        """model_service raises FileNotFoundError if artifact is missing."""
        from src.model_service import load_model
        import importlib, unittest.mock as mock
        with mock.patch("src.model_service.MODEL_PATH", tmp_path / "nonexistent.joblib"):
            with pytest.raises(FileNotFoundError):
                load_model()
