"""
tests/test_validation.py — Tests for input validation and sanitisation.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.validation import sanitise_listing, MAX_TEXT_LENGTH


class TestSanitiseListing:
    def test_valid_listing_no_errors(self):
        raw = {
            "title": "Software Intern",
            "description": "Build things",
            "email": "hr@example.com",
            "deadline": "2099-12-31",
            "telecommuting": 1,
            "has_company_logo": 0,
            "has_questions": 1,
        }
        clean, errors = sanitise_listing(raw)
        assert errors == []
        assert clean["title"] == "Software Intern"

    def test_empty_listing_requires_title_or_description(self):
        _, errors = sanitise_listing({})
        assert len(errors) == 1
        assert "title" in errors[0].lower() or "description" in errors[0].lower()

    def test_title_only_accepted(self):
        _, errors = sanitise_listing({"title": "Some Role"})
        assert errors == []

    def test_description_only_accepted(self):
        _, errors = sanitise_listing({"description": "Role details"})
        assert errors == []

    def test_invalid_email_adds_error(self):
        _, errors = sanitise_listing({"title": "Role", "email": "notanemail"})
        assert any("email" in e.lower() for e in errors)

    def test_valid_email_no_error(self):
        _, errors = sanitise_listing({"title": "Role", "email": "hr@example.com"})
        assert errors == []

    def test_invalid_date_adds_error(self):
        _, errors = sanitise_listing({"title": "Role", "deadline": "not-a-date"})
        assert any("deadline" in e.lower() for e in errors)

    def test_valid_iso_date_no_error(self):
        _, errors = sanitise_listing({"title": "Role", "deadline": "2099-06-30"})
        assert errors == []

    def test_text_field_truncated(self):
        long_text = "a" * (MAX_TEXT_LENGTH + 500)
        clean, _ = sanitise_listing({"title": "Role", "description": long_text})
        assert len(clean["description"]) == MAX_TEXT_LENGTH

    def test_none_values_handled_gracefully(self):
        raw = {
            "title": None, "company_name": None, "description": "A role",
            "telecommuting": None,
        }
        clean, errors = sanitise_listing(raw)
        assert clean["title"] == ""
        assert clean["telecommuting"] == 0

    def test_numeric_fields_converted(self):
        raw = {"title": "Role", "telecommuting": "1", "has_company_logo": "0", "has_questions": "1"}
        clean, errors = sanitise_listing(raw)
        assert clean["telecommuting"] == 1
        assert clean["has_company_logo"] == 0
        assert clean["has_questions"] == 1

    def test_all_expected_keys_present(self):
        clean, _ = sanitise_listing({"title": "Role"})
        expected_keys = [
            "title", "company_name", "company_profile", "website", "email", "phone",
            "description", "requirements", "benefits", "salary_claim", "salary_range",
            "deadline", "employment_type", "required_experience", "required_education",
            "industry", "function", "location", "telecommuting", "has_company_logo", "has_questions",
        ]
        for key in expected_keys:
            assert key in clean, f"Expected key missing from clean listing: {key}"
