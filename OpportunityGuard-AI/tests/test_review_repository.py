"""
tests/test_review_repository.py — Tests for the SQLite review queue.
All tests use temporary databases to avoid touching the real review_queue.db.
"""
import os
import sys
from pathlib import Path
import pytest
import tempfile
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.review_repository import (
    init_db,
    add_case,
    list_cases,
    get_case,
    update_decision,
    reopen_case,
    count_pending,
    find_pending_case,
    get_db_path,
    DEFAULT_DB_PATH,
    VALID_DECISIONS,
)

SAMPLE_LISTING = {
    "title": "Test Internship",
    "company_name": "Test Corp",
    "description": "A test listing for unit tests.",
    "email": "test@example.com",
}

SAMPLE_ASSESSMENT = {
    "composite_risk_score": 45,
    "model_risk_percent": 30.0,
    "rule_score": 20,
    "assessment": "Review recommended",
    "warnings": [{"code": "test_warning", "severity": "medium"}],
}


@pytest.fixture
def db_path(tmp_path):
    """Return a fresh temporary database path for each test."""
    db = tmp_path / "test_review.db"
    init_db(db)
    return db


class TestInitDb:
    def test_creates_file(self, tmp_path):
        db = tmp_path / "fresh.db"
        assert not db.exists()
        init_db(db)
        assert db.exists()

    def test_idempotent(self, db_path):
        """Calling init_db twice should not raise."""
        init_db(db_path)
        init_db(db_path)


class TestAddCase:
    def test_returns_integer_id(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        assert isinstance(cid, int)
        assert cid > 0

    def test_case_is_pending_by_default(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case["status"] == "PENDING"

    def test_listing_fields_stored(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case["listing_title"] == "Test Internship"
        assert case["company_name"] == "Test Corp"

    def test_scores_stored(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case["composite_score"] == 45
        assert case["model_risk_pct"] == 30.0
        assert case["rule_score"] == 20

    def test_listing_json_stored(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        case = get_case(cid, db_path=db_path)
        stored = json.loads(case["listing_json"])
        assert stored["title"] == "Test Internship"

    def test_deduplicate_true_returns_same_id(self, db_path):
        id1 = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=True)
        id2 = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=True)
        assert id1 == id2
        assert count_pending(db_path=db_path) == 1

    def test_deduplicate_false_creates_new_row(self, db_path):
        id1 = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=False)
        id2 = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=False)
        assert id1 != id2
        assert count_pending(db_path=db_path) == 2


class TestFindPendingCase:
    def test_find_existing_pending_case(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        found = find_pending_case(SAMPLE_LISTING, db_path=db_path)
        assert found is not None
        assert found["id"] == cid

    def test_find_pending_case_returns_none_if_no_match(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        diff_listing = dict(SAMPLE_LISTING, title="Different Title")
        assert find_pending_case(diff_listing, db_path=db_path) is None

    def test_find_pending_case_ignores_reviewed_cases(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        update_decision(cid, "LEGITIMATE", db_path=db_path)
        assert find_pending_case(SAMPLE_LISTING, db_path=db_path) is None


class TestListCases:
    def test_empty_db_returns_empty_list(self, db_path):
        assert list_cases(db_path=db_path) == []

    def test_returns_added_cases(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        cases = list_cases(db_path=db_path)
        assert len(cases) == 1

    def test_pending_filter(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        pending = list_cases(status_filter="PENDING", db_path=db_path)
        assert len(pending) == 1

    def test_reviewed_filter_empty_initially(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        reviewed = list_cases(status_filter="REVIEWED", db_path=db_path)
        assert len(reviewed) == 0

    def test_order_is_descending(self, db_path):
        id1 = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        l2 = dict(SAMPLE_LISTING, title="Second Listing")
        id2 = add_case(l2, SAMPLE_ASSESSMENT, db_path=db_path)
        cases = list_cases(db_path=db_path)
        assert cases[0]["id"] == id2
        assert cases[1]["id"] == id1


class TestGetCase:
    def test_existing_case(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case is not None
        assert case["id"] == cid

    def test_nonexistent_case_returns_none(self, db_path):
        assert get_case(9999, db_path=db_path) is None


class TestUpdateDecision:
    @pytest.mark.parametrize("decision", ["LEGITIMATE", "SUSPICIOUS", "INCONCLUSIVE"])
    def test_valid_decisions(self, db_path, decision):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        ok = update_decision(cid, decision, notes="Test notes", db_path=db_path)
        assert ok is True
        case = get_case(cid, db_path=db_path)
        assert case["status"] == "REVIEWED"
        assert case["reviewer_decision"] == decision
        assert case["reviewer_notes"] == "Test notes"
        assert case["reviewed_at"] is not None

    def test_invalid_decision_raises(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        with pytest.raises(ValueError):
            update_decision(cid, "INVALID_CHOICE", db_path=db_path)

    def test_nonexistent_case_returns_false(self, db_path):
        ok = update_decision(9999, "LEGITIMATE", db_path=db_path)
        assert ok is False

    def test_lowercase_decision_accepted(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        ok = update_decision(cid, "legitimate", db_path=db_path)
        assert ok is True
        case = get_case(cid, db_path=db_path)
        assert case["reviewer_decision"] == "LEGITIMATE"

    def test_decision_does_not_alter_original_ml_score(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        update_decision(cid, "LEGITIMATE", db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case["model_risk_pct"] == 30.0

    def test_subsequent_decision_update(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        update_decision(cid, "SUSPICIOUS", notes="First look", db_path=db_path)
        update_decision(cid, "LEGITIMATE", notes="Verified company", db_path=db_path)
        case = get_case(cid, db_path=db_path)
        assert case["status"] == "REVIEWED"
        assert case["reviewer_decision"] == "LEGITIMATE"
        assert case["reviewer_notes"] == "Verified company"

    def test_reopen_case(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        update_decision(cid, "SUSPICIOUS", db_path=db_path)
        assert count_pending(db_path=db_path) == 0
        reopened = reopen_case(cid, db_path=db_path)
        assert reopened is True
        assert count_pending(db_path=db_path) == 1
        case = get_case(cid, db_path=db_path)
        assert case["status"] == "PENDING"
        assert case["reviewer_decision"] is None


class TestCountPending:
    def test_empty_db_zero_pending(self, db_path):
        assert count_pending(db_path=db_path) == 0

    def test_one_added_one_pending(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        assert count_pending(db_path=db_path) == 1

    def test_after_review_zero_pending(self, db_path):
        cid = add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        update_decision(cid, "SUSPICIOUS", db_path=db_path)
        assert count_pending(db_path=db_path) == 0

    def test_multiple_cases(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path)
        assert count_pending(db_path=db_path) == 2

    def test_multiple_with_deduplication(self, db_path):
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=True)
        add_case(SAMPLE_LISTING, SAMPLE_ASSESSMENT, db_path=db_path, deduplicate=True)
        assert count_pending(db_path=db_path) == 1


class TestDatabasePathConsistency:
    def test_get_db_path_is_absolute(self):
        p = get_db_path()
        assert p.is_absolute()
        assert p.name == "review_queue.db"

    def test_get_db_path_consistent_across_cwd(self, tmp_path, monkeypatch):
        orig_path = get_db_path()
        monkeypatch.chdir(tmp_path)
        new_path = get_db_path()
        assert orig_path == new_path

    def test_env_override_db_path(self, tmp_path, monkeypatch):
        custom_db = tmp_path / "custom_queue.db"
        monkeypatch.setenv("OPPORTUNITYGUARD_DB_PATH", str(custom_db))
        p = get_db_path()
        assert p == custom_db.resolve()
