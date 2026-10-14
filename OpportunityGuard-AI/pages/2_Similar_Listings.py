"""
pages/2_Similar_Listings.py — Explore similar listings from the reference index.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="Similar Listings — OpportunityGuard AI",
    page_icon="🔄",
    layout="wide",
)

from src.resources import get_model_and_config, get_duplicate_index
from src.duplicate_detector import find_similar

st.title("🔄 Similar Listings Explorer")
st.caption(
    "Search the reference index of ~17,000 job descriptions for content similar to any text you provide. "
    "High similarity may indicate a copied or templated listing — investigate before applying."
)

with st.form("sim_search_form"):
    query_text = st.text_area(
        "Paste a job description to search for similar listings",
        height=200,
        placeholder="Paste the job or internship description here…",
        key="sim_query",
    )
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5, key="sim_k")
    with col2:
        threshold = st.slider(
            "Minimum similarity threshold",
            min_value=0.05,
            max_value=1.0,
            value=0.10,
            step=0.05,
            key="sim_threshold",
        )
    search_btn = st.form_submit_button("🔍 Find Similar Listings", use_container_width=True)

if search_btn:
    if not query_text.strip():
        st.error("Please enter a description to search.")
    else:
        with st.spinner("Searching index…"):
            try:
                dup_index = get_duplicate_index()
            except Exception as exc:
                st.error(f"Could not load duplicate index: {exc}")
                st.stop()

            results = find_similar(query_text, dup_index, top_k=top_k, threshold=threshold)

        if not results:
            st.info(f"No listings found with similarity ≥ {threshold:.0%}. Try lowering the threshold.")
        else:
            st.success(f"Found {len(results)} similar listing(s).")
            for i, r in enumerate(results, 1):
                sim_pct = r["similarity"] * 100
                if sim_pct >= 85:
                    icon = "🔴"
                    label = "Very High"
                elif sim_pct >= 50:
                    icon = "🟡"
                    label = "Moderate"
                else:
                    icon = "🔵"
                    label = "Low"

                with st.expander(
                    f"{icon} Match #{i} — {label} Similarity: {sim_pct:.1f}% (Index #{r['reference_index']})",
                    expanded=(i <= 3),
                ):
                    st.write(r["description_preview"])
                    st.caption(f"Reference index: {r['reference_index']} | Cosine similarity: {r['similarity']:.4f}")

            # High-similarity warning
            high_sim = [r for r in results if r["similarity"] >= 0.85]
            if high_sim:
                st.warning(
                    f"⚠️ {len(high_sim)} result(s) with ≥85% similarity detected. "
                    "This warrants careful investigation — the description may have been copied from an indexed listing."
                )

            # Disclaimer
            st.info(
                "**Similarity ≠ fraud.** Template descriptions, industry-standard language, and "
                "legitimate job boards often contain repeated phrases. High similarity is a signal "
                "to investigate, not proof of wrongdoing."
            )

# ─── Show last assessment similar listings if available ────────────────────────
if st.session_state.get("last_assessment"):
    assessment = st.session_state.last_assessment
    similar = assessment.get("similar_listings", [])
    if similar:
        st.divider()
        st.subheader("📋 Similar Listings from Last Assessment")
        for i, r in enumerate(similar, 1):
            sim_pct = r["similarity"] * 100
            with st.expander(f"Match #{i} — Similarity: {sim_pct:.1f}% (Index #{r['reference_index']})"):
                st.write(r["description_preview"])
