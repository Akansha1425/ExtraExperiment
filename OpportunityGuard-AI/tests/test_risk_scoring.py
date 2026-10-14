"""
tests/test_risk_scoring.py — Tests for composite risk score calculations.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.risk_scoring import (
    compute_composite_score,
    classify_composite,
    needs_human_review,
    build_full_assessment,
    CATEGORY_LOWER, CATEGORY_REVIEW, CATEGORY_HIGH,
    LOWER_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
)


class TestCompositeScore:
    def test_zero_inputs_zero_score(self):
        assert compute_composite_score(0.0, 0) == 0

    def test_formula_matches_config(self):
        # composite = round(0.65 * model_risk_percent + 0.35 * rule_score)
        assert compute_composite_score(50.0, 50) == round(0.65 * 50 + 0.35 * 50)

    def test_max_inputs_capped_at_100(self):
        assert compute_composite_score(100.0, 100) == 100

    def test_partial_ml_only(self):
        score = compute_composite_score(80.0, 0)
        assert score == round(0.65 * 80)

    def test_partial_rule_only(self):
        score = compute_composite_score(0.0, 100)
        assert score == round(0.35 * 100)

    def test_negative_values_clamped_to_zero(self):
        assert compute_composite_score(-10.0, -5) == 0

    def test_values_above_100_clamped_to_100(self):
        assert compute_composite_score(120.0, 110) == 100

    def test_unflagged_ml_45_with_rule_70_is_review_recommended_not_high_risk(self):
        # 0.65 * 45 + 0.35 * 70 = 29.25 + 24.5 = 53.75 -> round(53.75) = 54
        score = compute_composite_score(45.0, 70)
        assert score == 54
        assert classify_composite(score) == CATEGORY_REVIEW
        assert classify_composite(score) != CATEGORY_HIGH

    def test_ml_flagged_with_zero_rules_is_review_recommended_band(self):
        # 0.65 * 80 + 0.35 * 0 = 52.0 -> 52 (separate decisions: flagged ML, but Review Recommended composite tier)
        score = compute_composite_score(80.0, 0)
        assert score == 52
        assert classify_composite(score) == CATEGORY_REVIEW


class TestClassifyComposite:
    def test_zero_is_lower_risk(self):
        assert classify_composite(0) == CATEGORY_LOWER

    def test_29_is_lower_risk(self):
        assert classify_composite(LOWER_RISK_THRESHOLD - 1) == CATEGORY_LOWER

    def test_30_is_review_recommended(self):
        assert classify_composite(LOWER_RISK_THRESHOLD) == CATEGORY_REVIEW

    def test_59_is_review_recommended(self):
        assert classify_composite(HIGH_RISK_THRESHOLD - 1) == CATEGORY_REVIEW

    def test_60_is_high_risk(self):
        assert classify_composite(HIGH_RISK_THRESHOLD) == CATEGORY_HIGH

    def test_100_is_high_risk(self):
        assert classify_composite(100) == CATEGORY_HIGH


class TestNeedsHumanReview:
    def test_flagged_model_requires_review(self):
        assert needs_human_review(model_flagged=True, rule_score=0, composite_score=0) is True

    def test_high_rule_score_requires_review(self):
        assert needs_human_review(model_flagged=False, rule_score=35, composite_score=10) is True

    def test_high_composite_requires_review(self):
        assert needs_human_review(model_flagged=False, rule_score=0, composite_score=30) is True

    def test_all_low_no_review(self):
        assert needs_human_review(model_flagged=False, rule_score=10, composite_score=20) is False

    def test_borderline_rule_below_threshold(self):
        assert needs_human_review(model_flagged=False, rule_score=34, composite_score=29) is False

    def test_unflagged_ml_45_rule_70_routes_to_review_via_rule_and_composite(self):
        # rule_score 70 >= 35, composite 54 >= 30 -> requires review
        assert needs_human_review(model_flagged=False, rule_score=70, composite_score=54) is True


class TestBuildFullAssessment:
    def _make_model_result(self, score=0.1, flagged=False):
        return {
            "model_risk_score": score,
            "model_risk_percent": score * 100,
            "model_flagged": flagged,
            "model_threshold": 0.78,
            "model_version": "test-v1",
            "model_prediction": "Not flagged" if not flagged else "Flagged",
        }

    def _make_rule_result(self, score=0, warnings=None):
        return {
            "rule_score": score,
            "warnings": warnings or [],
        }

    def test_returns_dict(self):
        result = build_full_assessment({}, self._make_model_result(), self._make_rule_result(), [])
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = build_full_assessment({}, self._make_model_result(), self._make_rule_result(), [])
        for key in ("composite_risk_score", "assessment", "needs_human_review",
                    "warnings", "similar_listings", "formula", "disclaimer", "explanation"):
            assert key in result

    def test_formula_applied_correctly(self):
        result = build_full_assessment(
            {},
            self._make_model_result(score=0.5),
            self._make_rule_result(score=50),
            [],
        )
        expected = round(0.65 * 50 + 0.35 * 50)
        assert result["composite_risk_score"] == expected

    def test_disclaimer_present(self):
        result = build_full_assessment({}, self._make_model_result(), self._make_rule_result(), [])
        assert "disclaimer" in result
        assert len(result["disclaimer"]) > 10

    def test_similar_listings_passed_through(self):
        similar = [{"similarity": 0.9, "reference_index": 1, "description_preview": "test"}]
        result = build_full_assessment({}, self._make_model_result(), self._make_rule_result(), similar)
        assert result["similar_listings"] == similar

    def test_unflagged_ml_with_high_rules_full_assessment(self):
        # ML risk = 45% (flagged=False), rules = 95
        result = build_full_assessment(
            {},
            self._make_model_result(score=0.45, flagged=False),
            self._make_rule_result(score=95),
            [],
        )
        assert result["composite_risk_score"] == 62
        assert result["assessment"] == CATEGORY_HIGH
        assert result["needs_human_review"] is True


class TestRiskExplanation:
    def test_explanation_contains_formula_and_scores(self):
        from src.risk_scoring import generate_risk_explanation
        explanation = generate_risk_explanation(
            model_risk_percent=45.0,
            rule_score=70.0,
            composite_score=54,
            model_flagged=False,
            model_threshold=0.78,
        )
        assert "0.65" in explanation
        assert "0.35" in explanation
        assert "54" in explanation
        assert "Review recommended" in explanation
        assert "Separate Decisions" in explanation

    def test_explanation_cautious_wording(self):
        from src.risk_scoring import generate_risk_explanation
        explanation = generate_risk_explanation(
            model_risk_percent=85.0,
            rule_score=50.0,
            composite_score=73,
            model_flagged=True,
            model_threshold=0.78,
        )
        assert "not prove fraud" in explanation.lower() or "not proof of fraud" in explanation.lower()
        assert "risk indicator" in explanation.lower()

    def test_explanation_routing_triggers_described(self):
        from src.risk_scoring import generate_risk_explanation
        explanation = generate_risk_explanation(
            model_risk_percent=80.0,
            rule_score=40.0,
            composite_score=66,
            model_flagged=True,
            model_threshold=0.78,
        )
        assert "ML model flagged" in explanation
        assert "Rule score 40" in explanation
        assert "Composite score 66" in explanation
