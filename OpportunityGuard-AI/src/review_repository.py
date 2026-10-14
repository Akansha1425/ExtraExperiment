"""
review_repository.py — SQLite-backed human-review queue.

All SQL uses parameterised queries. The repository creates its own database
and table on first use. It is safe to call from Streamlit reruns; the
connection is short-lived per operation.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_DECISIONS = {"LEGITIMATE", "SUSPICIOUS", "INCONCLUSIVE"}


def get_db_path() -> Path:
    """Return the absolute, resolved path to the SQLite review database.

    Can be overridden via the OPPORTUNITYGUARD_DB_PATH environment variable
    or Streamlit secrets (database.path).
    Always returns a fully resolved, absolute Path to ensure consistency
    across pages and independent of the process's working directory.
    """
    env_override = os.environ.get("OPPORTUNITYGUARD_DB_PATH")
    if env_override:
        return Path(env_override).resolve()
    try:
        import streamlit as st
        secret_path = st.secrets.get("database", {}).get("path")
        if secret_path:
            return Path(secret_path).resolve()
    except Exception:
        pass
    return (Path(__file__).resolve().parent.parent / "review_queue.db").resolve()


DEFAULT_DB_PATH: Path = get_db_path()


def _connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path_str = str(db_path)
    if path_str != ":memory:":
        p = Path(path_str).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        path_str = str(p)
    conn = sqlite3.connect(path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create the review_cases table if it does not already exist."""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS review_cases (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT    NOT NULL,
                listing_title   TEXT,
                company_name    TEXT,
                composite_score INTEGER,
                model_risk_pct  REAL,
                rule_score      INTEGER,
                assessment      TEXT,
                listing_json    TEXT,
                warnings_json   TEXT,
                status          TEXT    NOT NULL DEFAULT 'PENDING',
                reviewer_decision TEXT,
                reviewer_notes  TEXT,
                reviewed_at     TEXT
            )
            """
        )
        conn.commit()
    logger.info("Review database initialised at %s", db_path)


def find_pending_case(
    listing: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Find an existing PENDING case matching the given listing.

    Matches by listing_title and company_name, and if description is present,
    verifies that the stored description matches.
    Returns the case dictionary if found, or None.
    """
    init_db(db_path)
    title = str(listing.get("title", ""))[:255].strip()
    company = str(listing.get("company_name", ""))[:255].strip()
    desc = str(listing.get("description", "")).strip()

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM review_cases
            WHERE status = 'PENDING' AND listing_title = ? AND company_name = ?
            ORDER BY id DESC
            """,
            (title, company),
        ).fetchall()

        for r in rows:
            case_dict = dict(r)
            if not desc:
                return case_dict
            try:
                stored_listing = json.loads(case_dict.get("listing_json", "{}"))
                if stored_listing.get("description", "").strip() == desc:
                    return case_dict
            except Exception:
                return case_dict
    return None


def add_case(
    listing: dict[str, Any],
    assessment: dict[str, Any],
    db_path: Path | str = DEFAULT_DB_PATH,
    deduplicate: bool = False,
) -> int:
    """Insert a new review case and return its row ID.

    If deduplicate=True and a PENDING case for this listing already exists,
    returns the existing case ID without creating a duplicate record.
    """
    init_db(db_path)
    if deduplicate:
        existing = find_pending_case(listing, db_path=db_path)
        if existing is not None:
            logger.info(
                "Found existing pending case id=%s for '%s'; reusing without duplicate",
                existing["id"],
                listing.get("title"),
            )
            return int(existing["id"])

    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO review_cases
              (created_at, listing_title, company_name, composite_score,
               model_risk_pct, rule_score, assessment,
               listing_json, warnings_json, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
            """,
            (
                now,
                str(listing.get("title", ""))[:255],
                str(listing.get("company_name", ""))[:255],
                int(assessment.get("composite_risk_score", 0)),
                float(assessment.get("model_risk_percent", 0.0)),
                int(assessment.get("rule_score", 0)),
                str(assessment.get("assessment", ""))[:255],
                json.dumps(listing, default=str),
                json.dumps(assessment.get("warnings", []), default=str),
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
    logger.info("Added review case id=%s", row_id)
    return row_id


def list_cases(
    status_filter: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return all review cases, optionally filtered by status."""
    init_db(db_path)
    with _connect(db_path) as conn:
        if status_filter:
            rows = conn.execute(
                "SELECT * FROM review_cases WHERE status = ? ORDER BY created_at DESC",
                (status_filter.upper(),),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM review_cases ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_case(case_id: int, db_path: Path | str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Fetch a single review case by ID."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM review_cases WHERE id = ?", (case_id,)
        ).fetchone()
    return dict(row) if row else None


def update_decision(
    case_id: int,
    decision: str,
    notes: str = "",
    db_path: Path | str = DEFAULT_DB_PATH,
) -> bool:
    """Record a reviewer decision. decision must be one of VALID_DECISIONS.

    Returns True on success, False if case not found.
    """
    decision = decision.upper().strip()
    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"Invalid decision '{decision}'. Must be one of: {VALID_DECISIONS}"
        )

    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        affected = conn.execute(
            """
            UPDATE review_cases
               SET status = 'REVIEWED',
                   reviewer_decision = ?,
                   reviewer_notes = ?,
                   reviewed_at = ?
             WHERE id = ?
            """,
            (decision, str(notes)[:2000], now, case_id),
        ).rowcount
        conn.commit()

    if affected == 0:
        logger.warning("update_decision: case id=%s not found", case_id)
        return False
    logger.info("Case id=%s updated: decision=%s", case_id, decision)
    return True


def reopen_case(case_id: int, db_path: Path | str = DEFAULT_DB_PATH) -> bool:
    """Reset a case back to PENDING status."""
    init_db(db_path)
    with _connect(db_path) as conn:
        affected = conn.execute(
            """
            UPDATE review_cases
               SET status = 'PENDING',
                   reviewer_decision = NULL,
                   reviewed_at = NULL
             WHERE id = ?
            """,
            (case_id,),
        ).rowcount
        conn.commit()
    return affected > 0


def count_pending(db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """Return the number of cases awaiting review."""
    init_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM review_cases WHERE status = 'PENDING'"
        ).fetchone()
    return row[0] if row else 0
