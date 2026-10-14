"""
validation.py — Input validation helpers for OpportunityGuard AI.

Validates and sanitises user-submitted listing data before it reaches
the model or rule engine.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

MAX_TEXT_LENGTH = 10_000   # chars per free-text field
MAX_TITLE_LENGTH = 300

EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def _truncate(value: Any, max_len: int) -> str:
    """Return a string, truncated to max_len chars."""
    s = str(value or "").strip()
    return s[:max_len]


def sanitise_listing(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Sanitise and validate a raw listing dict.

    Returns:
        (clean_listing, validation_errors)

    validation_errors is empty when the listing is acceptable.
    The clean_listing always contains all expected keys (with empty defaults
    for missing optional fields), so downstream code does not need to guard
    every key access.
    """
    errors: list[str] = []

    # Required: title or description
    title = _truncate(raw.get("title", ""), MAX_TITLE_LENGTH)
    description = _truncate(raw.get("description", ""), MAX_TEXT_LENGTH)
    if not title and not description:
        errors.append("Provide at least a job title or a job description.")

    # Email format (optional field)
    email = _truncate(raw.get("email", ""), 254)
    if email and not EMAIL_RE.match(email):
        errors.append(f"Email address '{email}' does not look valid.")

    # Deadline (optional field)
    deadline_str = _truncate(raw.get("deadline", ""), 30)
    if deadline_str:
        try:
            date.fromisoformat(deadline_str)
        except ValueError:
            errors.append(f"Deadline '{deadline_str}' is not a valid ISO date (YYYY-MM-DD).")

    # Numeric fields
    def _to_int_flag(val: Any, name: str) -> int:
        if val is None or val == "":
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            errors.append(f"Field '{name}' must be 0 or 1.")
            return 0

    telecommuting = _to_int_flag(raw.get("telecommuting", 0), "telecommuting")
    has_company_logo = _to_int_flag(raw.get("has_company_logo", 0), "has_company_logo")
    has_questions = _to_int_flag(raw.get("has_questions", 0), "has_questions")

    clean: dict[str, Any] = {
        "title": title,
        "company_name": _truncate(raw.get("company_name", ""), 255),
        "company_profile": _truncate(raw.get("company_profile", ""), MAX_TEXT_LENGTH),
        "website": _truncate(raw.get("website", ""), 500),
        "email": email,
        "phone": _truncate(raw.get("phone", ""), 50),
        "description": description,
        "requirements": _truncate(raw.get("requirements", ""), MAX_TEXT_LENGTH),
        "benefits": _truncate(raw.get("benefits", ""), MAX_TEXT_LENGTH),
        "salary_claim": _truncate(raw.get("salary_claim", ""), 300),
        "salary_range": _truncate(raw.get("salary_range", ""), 100),
        "deadline": deadline_str,
        "employment_type": _truncate(raw.get("employment_type", ""), 100),
        "required_experience": _truncate(raw.get("required_experience", ""), 100),
        "required_education": _truncate(raw.get("required_education", ""), 100),
        "industry": _truncate(raw.get("industry", ""), 100),
        "function": _truncate(raw.get("function", ""), 100),
        "location": _truncate(raw.get("location", ""), 200),
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
    }

    return clean, errors
