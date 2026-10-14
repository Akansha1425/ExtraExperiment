"""
pages/3_Review_Queue.py — Human-review queue for OpportunityGuard AI.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Review Queue — OpportunityGuard AI",
    page_icon="📋",
    layout="wide",
)

from src.review_repository import (
    list_cases,
    get_case,
    update_decision,
    reopen_case,
    count_pending,
    DEFAULT_DB_PATH,
    VALID_DECISIONS,
    init_db,
)

st.title("📋 Human Review Queue")
st.caption(
    "Review flagged listings, record your findings, and mark cases as "
    "LEGITIMATE, SUSPICIOUS, or INCONCLUSIVE. "
    "**Note:** On Streamlit Community Cloud the SQLite database is stored in "
    "ephemeral deployment storage and will not survive redeployments."
)

# Ensure DB is initialised
init_db(DEFAULT_DB_PATH)

# ─── Filter controls ───────────────────────────────────────────────────────────
col_filter, col_refresh = st.columns([3, 1])
with col_filter:
    status_filter = st.selectbox(
        "Filter by status",
        ["All", "PENDING", "REVIEWED"],
        key="status_filter",
    )
with col_refresh:
    st.write("")
    if st.button("🔄 Refresh", key="refresh_btn", use_container_width=True):
        st.rerun()

filter_val = None if status_filter == "All" else status_filter
cases = list_cases(status_filter=filter_val, db_path=DEFAULT_DB_PATH)

pending_count = count_pending(DEFAULT_DB_PATH)
st.metric("Pending Reviews", pending_count)

if not cases:
    st.info(
        "No cases in the queue yet. Run an assessment and click "
        "**Add to Human Review Queue** to add a case."
    )
    st.stop()

st.divider()
st.subheader(f"Cases ({len(cases)} shown)")

# ─── Case list ─────────────────────────────────────────────────────────────────
for case in cases:
    cid = case["id"]
    status = case.get("status", "PENDING")
    composite = case.get("composite_score", 0)
    title_display = case.get("listing_title") or "Untitled"
    company_display = case.get("company_name") or "Unknown company"

    status_icon = "⏳" if status == "PENDING" else "✅"
    if composite >= 60:
        risk_badge = "🔴"
    elif composite >= 30:
        risk_badge = "🟡"
    else:
        risk_badge = "🟢"

    expander_label = (
        f"{status_icon} Case #{cid} — {risk_badge} {composite}/100 — "
        f"{title_display} @ {company_display} [{status}]"
    )

    with st.expander(expander_label, expanded=status == "PENDING"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Composite Score", f"{composite}/100")
        c2.metric("ML Risk", f"{case.get('model_risk_pct', 0):.1f}%")
        c3.metric("Rule Score", f"{case.get('rule_score', 0)}/100")

        st.write(f"**Assessment:** {case.get('assessment', '')}")
        st.write(f"**Created:** {case.get('created_at', '')}")
        if status == "REVIEWED":
            st.success(
                f"✅ Reviewed — Decision: **{case.get('reviewer_decision', '')}** "
                f"at {case.get('reviewed_at', '')}"
            )
            if case.get("reviewer_notes"):
                st.write(f"**Notes:** {case.get('reviewer_notes', '')}")

        # Warnings
        warnings_raw = case.get("warnings_json", "[]")
        try:
            warnings = json.loads(warnings_raw)
        except Exception:
            warnings = []
        if warnings:
            with st.expander(f"⚠️ Triggered Warnings ({len(warnings)})", expanded=False):
                for w in warnings:
                    sev = w.get("severity", "low")
                    icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}.get(sev, "⚠️")
                    st.write(
                        f"{icon} **{w.get('code', '')}** [{sev}] — {w.get('evidence', '')}"
                    )
                    st.caption(f"→ {w.get('recommendation', '')}")

        # Original listing JSON
        listing_raw = case.get("listing_json", "{}")
        try:
            listing_data = json.loads(listing_raw)
        except Exception:
            listing_data = {}
        with st.expander("📄 Original Listing Data", expanded=False):
            st.json(listing_data)

        # Decision form (for both recording and updating decisions)
        st.divider()
        st.subheader("📝 Record Decision" if status == "PENDING" else "✏️ Update Decision / Notes")
        decision_key = f"decision_{cid}"
        notes_key = f"notes_{cid}"
        submit_key = f"submit_{cid}"

        options = ["", "LEGITIMATE", "SUSPICIOUS", "INCONCLUSIVE"]
        current_decision = case.get("reviewer_decision") or ""
        default_index = options.index(current_decision) if current_decision in options else 0

        decision = st.selectbox(
            "Decision",
            options,
            index=default_index,
            key=decision_key,
        )
        notes = st.text_area(
            "Reviewer Notes (optional)",
            value=case.get("reviewer_notes") or "",
            key=notes_key,
            placeholder="Add your observations, evidence checked, or rationale…",
            height=80,
        )

        col_act1, col_act2 = st.columns([2, 1])
        with col_act1:
            btn_label = "✅ Submit Decision" if status == "PENDING" else "🔄 Update Decision"
            if st.button(btn_label, key=submit_key, use_container_width=True):
                if not decision:
                    st.error("Please select a decision.")
                else:
                    try:
                        ok = update_decision(
                            case_id=cid,
                            decision=decision,
                            notes=notes,
                            db_path=DEFAULT_DB_PATH,
                        )
                        if ok:
                            st.success(f"Case #{cid} marked as **{decision}**.")
                            st.rerun()
                        else:
                            st.error(f"Case #{cid} not found.")
                    except ValueError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        st.error(f"Database error: {exc}")

        with col_act2:
            if status == "REVIEWED":
                if st.button("↩️ Reopen as PENDING", key=f"reopen_{cid}", use_container_width=True):
                    reopen_case(cid, db_path=DEFAULT_DB_PATH)
                    st.rerun()
