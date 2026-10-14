"""
pages/1_Assess_Listing.py — Full listing assessment form.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Assess Listing — OpportunityGuard AI",
    page_icon="🔍",
    layout="wide",
)

# Reuse cached resources from app.py session
from src.resources import get_model_and_config, get_duplicate_index
from src.validation import sanitise_listing
from src.model_service import predict, build_model_row
from src.rule_engine import run_all_rules
from src.duplicate_detector import find_similar
from src.risk_scoring import build_full_assessment, generate_risk_explanation
from src.review_repository import add_case, find_pending_case, DEFAULT_DB_PATH
from src.report_export import to_json, to_csv

# ─── Page styles ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.score-box {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #334155;
    text-align: center;
    margin-bottom: 0.5rem;
}
.score-title { color: #94a3b8; font-size: 0.85rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
.score-value { font-size: 2.5rem; font-weight: 700; line-height: 1.1; }
.score-green  { color: #4ade80; }
.score-orange { color: #fb923c; }
.score-red    { color: #f87171; }
.warning-card { background: #1c1917; border-left: 4px solid #f59e0b; border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; margin: 0.4rem 0; }
.w-critical   { border-left-color: #ef4444; }
.w-high       { border-left-color: #f97316; }
.w-medium     { border-left-color: #eab308; }
.w-low        { border-left-color: #3b82f6; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔍 Assess a Listing")
st.caption(
    "Fill in the listing details below and click **Run Assessment**. "
    "All fields except title or description are optional."
)

# ─── Quick-paste fallback ──────────────────────────────────────────────────────
with st.expander("📋 Quick Paste — paste the full listing text here (optional)", expanded=False):
    paste_text = st.text_area(
        "Paste listing text",
        height=150,
        placeholder="Paste the full job/internship listing here. The text will be used as the description.",
        key="paste_text",
    )

# ─── Main form ─────────────────────────────────────────────────────────────────
with st.form("listing_form", clear_on_submit=False):
    st.subheader("📋 Listing Details")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job / Internship Title *", placeholder="e.g. Software Development Intern", key="f_title")
        company_name = st.text_input("Company Name", placeholder="e.g. Acme Tech Ltd", key="f_company")
        website = st.text_input("Company Website", placeholder="https://acme.example.com", key="f_website")
        email = st.text_input("Employer Email", placeholder="hr@acme.example.com", key="f_email")
        phone = st.text_input("Phone Number", placeholder="+91 98765 43210", key="f_phone")
        deadline = st.date_input("Application Deadline", value=None, key="f_deadline")

    with col2:
        employment_type = st.selectbox(
            "Employment Type",
            ["", "Full-time", "Part-time", "Contract", "Temporary", "Internship", "Other"],
            key="f_emp_type",
        )
        required_experience = st.selectbox(
            "Required Experience",
            ["", "Not Applicable", "Internship", "Entry level", "Associate",
             "Mid-Senior level", "Director", "Executive"],
            key="f_experience",
        )
        required_education = st.selectbox(
            "Required Education",
            ["", "Unspecified", "High School or equivalent", "Some College Coursework Completed",
             "Vocational", "Associate Degree", "Bachelor's Degree", "Master's Degree",
             "Doctorate", "Professional", "Certification"],
            key="f_education",
        )
        industry = st.text_input("Industry", placeholder="e.g. Information Technology", key="f_industry")
        function = st.text_input("Function / Department", placeholder="e.g. Engineering", key="f_function")
        location = st.text_input("Location", placeholder="e.g. New Delhi, India or Remote", key="f_location")

    st.subheader("📝 Description & Details")
    company_profile = st.text_area("Company Profile", height=80, key="f_profile",
                                   placeholder="Brief description of the company…")
    description = st.text_area("Job Description *", height=150, key="f_desc",
                                placeholder="Describe the role, responsibilities…")
    requirements = st.text_area("Requirements", height=100, key="f_req",
                                 placeholder="Skills, qualifications required…")
    benefits = st.text_area("Benefits / Perks", height=80, key="f_benefits",
                             placeholder="Salary, perks, insurance…")

    st.subheader("💰 Compensation")
    col3, col4 = st.columns(2)
    with col3:
        salary_claim = st.text_input("Salary / Stipend Claim", placeholder="e.g. ₹15,000/month", key="f_salary_claim")
    with col4:
        salary_range = st.text_input("Salary Range (if available)", placeholder="e.g. 10000-20000", key="f_salary_range")

    st.subheader("⚙️ Additional Information")
    col5, col6, col7 = st.columns(3)
    with col5:
        telecommuting = st.checkbox("Remote / Telecommuting", key="f_telecommute")
    with col6:
        has_company_logo = st.checkbox("Has Company Logo", key="f_logo")
    with col7:
        has_questions = st.checkbox("Has Screening Questions", key="f_questions")

    submitted = st.form_submit_button("🔍 Run Assessment", use_container_width=True)

# ─── Assessment logic ──────────────────────────────────────────────────────────
if submitted:
    # Build raw listing dict
    raw = {
        "title": title,
        "company_name": company_name,
        "company_profile": company_profile,
        "website": website,
        "email": email,
        "phone": phone,
        "description": description if description else paste_text,
        "requirements": requirements,
        "benefits": benefits,
        "salary_claim": salary_claim,
        "salary_range": salary_range,
        "deadline": str(deadline) if deadline else "",
        "employment_type": employment_type,
        "required_experience": required_experience,
        "required_education": required_education,
        "industry": industry,
        "function": function,
        "location": location,
        "telecommuting": int(telecommuting),
        "has_company_logo": int(has_company_logo),
        "has_questions": int(has_questions),
    }

    listing, errors = sanitise_listing(raw)

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    else:
        with st.spinner("Running assessment…"):
            try:
                pipeline, config = get_model_and_config()
                dup_index = get_duplicate_index()
            except Exception as exc:
                st.error(f"Model resources unavailable: {exc}")
                st.stop()

            # ML prediction
            model_result = predict(listing, pipeline, config)

            # Rule engine
            rule_result = run_all_rules(listing)

            # Duplicate detection
            similar = find_similar(
                listing.get("description", ""),
                dup_index,
                top_k=5,
                threshold=0.15,  # show even low similarity for transparency
            )

            # Build full assessment
            assessment = build_full_assessment(listing, model_result, rule_result, similar)

        # Store in session history (avoid duplicates on rerun)
        history = st.session_state.get("assessment_history", [])
        if not history or history[-1].get("_listing_title") != listing.get("title"):
            record = dict(assessment)
            record["_listing_title"] = listing.get("title", "")
            history.append(record)
            st.session_state.assessment_history = history

        # Update current listing & assessment in session
        st.session_state.current_listing = listing
        st.session_state.current_assessment = assessment
        st.session_state.last_listing = listing
        st.session_state.last_assessment = assessment
        # Reset review queue submitted state for new assessment
        st.session_state.review_submitted_case_id = None

# ─── Results display (persists across reruns and button clicks) ───────────────
current_listing = st.session_state.get("current_listing")
current_assessment = st.session_state.get("current_assessment")

if current_listing and current_assessment:
    st.divider()
    st.subheader("📊 Assessment Results")

    composite = current_assessment["composite_risk_score"]
    model_pct = current_assessment["model_risk_percent"]
    rule_score = current_assessment["rule_score"]
    category = current_assessment["assessment"]

    if composite < 30:
        score_class = "score-green"
        badge_emoji = "🟢"
    elif composite < 60:
        score_class = "score-orange"
        badge_emoji = "🟡"
    else:
        score_class = "score-red"
        badge_emoji = "🔴"

    # Score cards
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f'<div class="score-box">'
            f'<div class="score-title">ML Risk Score</div>'
            f'<div class="score-value {score_class}">{model_pct:.1f}%</div>'
            f'<div style="color:#64748b;font-size:0.8rem">threshold: {current_assessment["model_threshold"]:.0%}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f'<div class="score-box">'
            f'<div class="score-title">Rule-Based Score</div>'
            f'<div class="score-value {score_class}">{rule_score}/100</div>'
            f'<div style="color:#64748b;font-size:0.8rem">deterministic checks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f'<div class="score-box">'
            f'<div class="score-title">Composite Risk Score</div>'
            f'<div class="score-value {score_class}">{composite}/100</div>'
            f'<div style="color:#64748b;font-size:0.8rem">{current_assessment["formula"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Progress bar
    st.progress(min(composite / 100, 1.0))
    st.markdown(f"**{badge_emoji} {category}**")
    st.caption(f"⚠️ {current_assessment['disclaimer']}")

    # ML prediction detail
    with st.expander("🤖 ML Model Detail & Risk Band Logic", expanded=False):
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.write(f"**ML Binary Flag:** {current_assessment['model_prediction']}")
            st.write(f"**Raw Model Score:** {current_assessment['model_risk_score']:.4f} ({current_assessment['model_risk_percent']:.1f}%)")
            st.write(f"**ML Decision Threshold:** {current_assessment['model_threshold']:.2f} (78.0%)")
            st.write(f"**Model Version:** `{current_assessment['model_version']}`")
        with c_m2:
            st.write(f"**Rule Score:** {current_assessment['rule_score']} / 100")
            st.write(f"**Composite Score:** {current_assessment['composite_risk_score']} / 100")
            st.write(f"**Composite Category:** `{current_assessment['assessment']}`")
            st.write(f"**Needs Human Review:** `{current_assessment['needs_human_review']}`")

        # Render dynamic risk scoring breakdown and explanation
        explanation = current_assessment.get("explanation")
        if not explanation:
            explanation = generate_risk_explanation(
                model_risk_percent=current_assessment["model_risk_percent"],
                rule_score=current_assessment["rule_score"],
                composite_score=current_assessment["composite_risk_score"],
                model_flagged=current_assessment.get("model_flagged", False),
                model_threshold=current_assessment.get("model_threshold", 0.78),
            )
        st.markdown(explanation)
        st.info(
            "⚠️ **Important Note:** The ML model score is derived from lexical patterns and metadata learned during training. "
            "It is a heuristic risk indicator, NOT a calibrated empirical probability. A high score is not definitive proof of fraud, "
            "and a low score does not guarantee safety. Always perform independent verification."
        )

    # Warnings
    warnings = current_assessment.get("warnings", [])
    if warnings:
        st.subheader(f"⚠️ Triggered Warnings ({len(warnings)})")
        SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        for w in sorted(warnings, key=lambda x: SEV_ORDER.get(x.get("severity", "low"), 9)):
            sev = w.get("severity", "low")
            card_class = f"w-{sev}"
            sev_icon = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🔵"}.get(sev, "⚠️")
            with st.expander(
                f"{sev_icon} [{sev.upper()} +{w.get('points', 0)} pts] {w.get('code', '')}",
                expanded=sev in ("critical", "high"),
            ):
                st.write(f"**Field:** `{w.get('field', '')}`")
                st.write(f"**Evidence:** {w.get('evidence', '')}")
                st.write(f"**Recommendation:** {w.get('recommendation', '')}")
    else:
        st.success("✅ No rule-based warnings triggered.")

    # Similar listings
    similar_list = current_assessment.get("similar_listings", [])
    if similar_list:
        st.subheader("🔄 Similar Indexed Descriptions")
        st.caption(
            "Descriptions are compared to a reference index of ~17,000 listings. "
            "High similarity warrants investigation — it does not confirm copying."
        )
        for i, s in enumerate(similar_list, 1):
            sim_pct = s["similarity"] * 100
            color = "🔴" if sim_pct >= 85 else ("🟡" if sim_pct >= 50 else "🔵")
            with st.expander(f"{color} Match #{i} — Similarity: {sim_pct:.1f}% (index #{s['reference_index']})"):
                st.write(s["description_preview"])
                if sim_pct >= 85:
                    st.warning("⚠️ Very high similarity — investigate whether this listing has been copied.")
    else:
        st.info("No closely similar listings found in the reference index.")

    # ─── Human review queue ───────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Human Review Queue")
    needs_review = current_assessment.get("needs_human_review", False)
    if needs_review:
        st.warning("🔎 This listing meets the criteria for human review.")
    else:
        st.info("ℹ️ This listing did not automatically trigger high-risk review criteria, but can still be submitted for manual review.")

    # Check whether this listing is already queued
    submitted_id = st.session_state.get("review_submitted_case_id")
    if not submitted_id:
        existing_case = find_pending_case(current_listing, db_path=DEFAULT_DB_PATH)
        if existing_case:
            submitted_id = existing_case["id"]
            st.session_state.review_submitted_case_id = submitted_id

    if submitted_id:
        st.success(f"✅ Listing is queued for human review as **Case #{submitted_id}**.")
        st.caption("You can view, inspect, and update reviewer decisions on the **Review Queue** page in the sidebar.")
    else:
        btn_label = "🚨 Add to Human Review Queue" if needs_review else "📋 Add to Human Review Queue"
        if st.button(btn_label, key="add_review_btn", use_container_width=True):
            try:
                new_id = add_case(
                    current_listing,
                    current_assessment,
                    db_path=DEFAULT_DB_PATH,
                    deduplicate=True,
                )
                st.session_state.review_submitted_case_id = new_id
                st.success(f"✅ Successfully added to human review queue as **Case #{new_id}**!")
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Failed to add case to review queue: {exc}")

    # ─── Export ───────────────────────────────────────────────────────────────
    st.subheader("📤 Export Report")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        json_str = to_json(current_assessment, current_listing)
        st.download_button(
            "⬇️ Download JSON",
            data=json_str,
            file_name=f"opportunityguard_assessment_{current_listing.get('title','listing')[:30]}.json",
            mime="application/json",
            key="dl_json_btn",
        )
    with col_dl2:
        csv_str = to_csv(current_assessment, current_listing)
        st.download_button(
            "⬇️ Download CSV",
            data=csv_str,
            file_name=f"opportunityguard_assessment_{current_listing.get('title','listing')[:30]}.csv",
            mime="text/csv",
            key="dl_csv_btn",
        )
