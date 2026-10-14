"""
app.py — OpportunityGuard AI: Streamlit entry point (Dashboard page).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="OpportunityGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/",
        "Report a bug": None,
        "About": "OpportunityGuard AI — ML-powered job listing safety checker",
    },
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero-box {
        background: linear-gradient(135deg, #1a1f3a 0%, #0f3460 50%, #16213e 100%);
        border-radius: 16px; padding: 2.5rem 2rem; margin-bottom: 1.5rem;
        border: 1px solid #1e3a5f;
    }
    .hero-title {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin: 0;
    }
    .hero-subtitle { color: #94a3b8; font-size: 1.1rem; margin-top: 0.5rem; }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b, #162032);
        border-radius: 12px; border: 1px solid #334155; padding: 1rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #1a1f2e 100%);
        border-right: 1px solid #1e293b;
    }
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.6rem 1.5rem; transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }
    [data-testid="stExpander"] {
        background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] { background: #0f1117; gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e293b; border-radius: 8px 8px 0 0;
        color: #94a3b8; font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1d4ed8, #6d28d9); color: white;
    }
    h2 { color: #e2e8f0; }
    h3 { color: #cbd5e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Startup validation ────────────────────────────────────────────────────────
_artifacts_dir = ROOT / "artifacts"
_required = {
    "opportunityguard_model.joblib": "Trained ML pipeline",
    "opportunityguard_duplicate_index.joblib": "Duplicate detection index",
    "opportunityguard_config.json": "Model configuration",
}
_missing = {name: desc for name, desc in _required.items()
            if not (_artifacts_dir / name).exists()}

if _missing:
    st.error(
        "**Missing required artifacts in `artifacts/`:**\n\n"
        + "\n".join(f"- `{name}` — {desc}" for name, desc in _missing.items())
        + "\n\nCopy the exported Colab artifacts into `artifacts/` and restart."
    )
    st.stop()

# ─── Load cached resources ─────────────────────────────────────────────────────
from src.resources import get_model_and_config, get_duplicate_index

try:
    _pipeline, _config = get_model_and_config()
    _dup_index = get_duplicate_index()
except Exception as exc:
    st.error(f"**Failed to load model resources:** {exc}")
    st.stop()

# ─── Session state init ────────────────────────────────────────────────────────
if "assessment_history" not in st.session_state:
    st.session_state.assessment_history = []

# ─── Dashboard ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-box">
      <p class="hero-title">🛡️ OpportunityGuard AI</p>
      <p class="hero-subtitle">
        ML-powered safety assessment for internship, job, and freelance listings.
        Identify suspicious patterns and verify employers before you apply.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

from src.review_repository import count_pending

history = st.session_state.assessment_history
pending = count_pending()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Assessments (session)", len(history))
with c2:
    high_risk = sum(1 for a in history if a.get("composite_risk_score", 0) >= 60)
    st.metric("High-Risk Detected", high_risk)
with c3:
    st.metric("Pending Human Reviews", pending)
with c4:
    st.metric("Model Version", _config.get("model_version", "—"))

st.divider()

# ─── Model performance metrics ─────────────────────────────────────────────────
metrics_path = _artifacts_dir / "final_test_metrics.csv"
if metrics_path.exists():
    import pandas as pd
    import plotly.graph_objects as go

    st.subheader("📊 Model Performance (Held-out Test Set)")
    st.caption(
        "Metrics from the Logistic Regression model on a held-out test set. "
        "These describe historical performance, not predictions for any individual listing."
    )
    df_m = pd.read_csv(metrics_path)
    row = df_m.iloc[0]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy", f"{row['Accuracy']:.1%}")
    mc2.metric("Balanced Accuracy", f"{row['Balanced accuracy']:.1%}")
    mc3.metric("Fraud Precision", f"{row['Fraud precision']:.1%}")
    mc4.metric("Fraud Recall", f"{row['Fraud recall']:.1%}")
    mc5.metric("F1 Score", f"{row['Fraud F1']:.1%}")

    mc6, mc7, mc8, mc9, mc10 = st.columns(5)
    mc6.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}")
    mc7.metric("PR-AUC", f"{row['Average precision / PR-AUC']:.4f}")
    mc8.metric("MCC", f"{row['MCC']:.4f}")
    mc9.metric("Brier Score", f"{row['Brier score']:.4f}")
    mc10.metric("Log Loss", f"{row['Log loss']:.4f}")

    # Radar chart
    labels = ["Accuracy", "Bal. Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    vals = [
        row["Accuracy"], row["Balanced accuracy"], row["Fraud precision"],
        row["Fraud recall"], row["Fraud F1"], row["ROC-AUC"],
        row["Average precision / PR-AUC"],
    ]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        fill="toself", fillcolor="rgba(99,102,241,0.2)",
        line=dict(color="#6366f1", width=2), name="Performance",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155", color="#94a3b8"),
            angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
        ),
        paper_bgcolor="#0f1117", font=dict(color="#e2e8f0", family="Inter"),
        showlegend=False, height=350, margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ─── Session risk distribution ─────────────────────────────────────────────────
if history:
    import plotly.express as px
    st.subheader("📈 Session Assessment Distribution")
    cats = []
    for a in history:
        s = a.get("composite_risk_score", 0)
        cats.append("High Risk" if s >= 60 else ("Review Recommended" if s >= 30 else "Lower Risk"))
    cat_counts = {c: cats.count(c) for c in ["Lower Risk", "Review Recommended", "High Risk"]}
    fig2 = px.pie(
        names=list(cat_counts.keys()), values=list(cat_counts.values()),
        color=list(cat_counts.keys()),
        color_discrete_map={"Lower Risk": "#4ade80", "Review Recommended": "#fb923c", "High Risk": "#f87171"},
        hole=0.5,
    )
    fig2.update_layout(
        paper_bgcolor="#0f1117", font=dict(color="#e2e8f0", family="Inter"),
        legend=dict(orientation="h"), height=280, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

# ─── Dataset info ──────────────────────────────────────────────────────────────
schema_path = _artifacts_dir / "dataset_schema.json"
if schema_path.exists():
    with schema_path.open() as f:
        schema = json.load(f)
    st.subheader("📂 Training Dataset")
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Total Records (deduped)", f"{schema.get('rows_after_deduplication', 0):,}")
    dc2.metric("Fraudulent Listings", f"{schema.get('fraudulent_count', 0):,}")
    dc3.metric("Legitimate Listings", f"{schema.get('legitimate_count', 0):,}")
    st.caption(
        "Source: [EMSCAD — Real or Fake Job Postings (Kaggle)]"
        "(https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)"
    )
    st.divider()

# ─── Capabilities & limitations ───────────────────────────────────────────────
st.subheader("🔍 System Capabilities")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**✅ Detects:**
- Upfront payment language (fees, deposits)
- Missing employer information
- Unrealistic salary/earnings claims
- Expired or invalid deadlines
- Email/website domain inconsistencies
- High textual similarity to indexed listings

**📊 Model (Logistic Regression):**
- 17,880 training records | ~5% fraud rate
- Test ROC-AUC: **0.990** | F1: **0.835** | MCC: **0.832**
    """)
with col_b:
    st.markdown("""
**⚠️ Limitations:**
- A flagged listing is **not confirmed fraud**
- A clean score does **not guarantee safety**
- No real-time employer or website verification
- Pattern matching can be evaded by paraphrasing
- Composite score is **not a calibrated probability**
- Always verify employers independently

**🔒 Privacy:**
- No listing data sent to external servers
- All processing is local to this deployment
    """)

st.info("👈 Use the **sidebar** to navigate to **Assess a Listing**, **Similar Listings**, **Review Queue**, **Model Evaluation**, or **About**.")
