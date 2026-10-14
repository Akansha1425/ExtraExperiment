"""
pages/5_About.py — Project documentation and limitations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(
    page_title="About — OpportunityGuard AI",
    page_icon="ℹ️",
    layout="wide",
)

st.title("ℹ️ About OpportunityGuard AI")

# ─── Project overview ──────────────────────────────────────────────────────────
st.markdown("""
## Project Purpose

**OpportunityGuard AI** is an ML-powered safety assessment tool for students evaluating
internship, job, and freelance listings. It combines a trained machine-learning model with
a deterministic rule engine to identify suspicious patterns and provide transparent,
explainable risk indicators.

The system was built as a college capstone/demonstration project following sound production
engineering practices: real trained artifacts, explainable scoring, a human-review workflow,
and honest disclaimers about its limitations.
""")

st.divider()

# ─── Architecture ──────────────────────────────────────────────────────────────
st.subheader("🏗️ System Architecture")
st.markdown("""
```
OpportunityGuard-AI/
├── app.py                    ← Dashboard (entry point: streamlit run app.py)
├── pages/
│   ├── 1_Assess_Listing.py   ← Full listing form + ML+rule assessment
│   ├── 2_Similar_Listings.py ← Cosine similarity search
│   ├── 3_Review_Queue.py     ← Human-review SQLite workflow
│   ├── 4_Model_Evaluation.py ← Real metrics and charts
│   └── 5_About.py            ← This page
├── src/
│   ├── model_service.py      ← Loads joblib pipeline, runs predict_proba
│   ├── rule_engine.py        ← Payment, employer, deadline, contact checks
│   ├── duplicate_detector.py ← TF-IDF cosine similarity against index
│   ├── risk_scoring.py       ← Composite score formula
│   ├── review_repository.py  ← SQLite CRUD for review cases
│   ├── validation.py         ← Input sanitisation and validation
│   └── report_export.py      ← JSON/CSV export
├── artifacts/                ← Model, duplicate index, config, metrics
├── tests/                    ← pytest unit tests
└── .streamlit/config.toml    ← Streamlit server configuration
```
""")

st.divider()

# ─── Dataset ───────────────────────────────────────────────────────────────────
st.subheader("📂 Dataset")
schema_path = ROOT / "artifacts" / "dataset_schema.json"
if schema_path.exists():
    with schema_path.open() as f:
        schema = json.load(f)
    st.markdown(f"""
- **Source:** [EMSCAD — Real or Fake Job Postings (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Total records (after deduplication):** {schema.get('rows_after_deduplication', 0):,}
- **Fraudulent listings:** {schema.get('fraudulent_count', 0):,} ({schema.get('fraudulent_count',0)/schema.get('rows_after_deduplication',1)*100:.1f}%)
- **Legitimate listings:** {schema.get('legitimate_count', 0):,}
- **Class imbalance:** The dataset is heavily imbalanced (~5% fraud).
  The model threshold was tuned (0.78) to trade some recall for higher precision.
""")

st.divider()

# ─── Model training ───────────────────────────────────────────────────────────
st.subheader("🤖 Model Training Methodology")
st.markdown("""
**Feature engineering:**
- All text fields (title, description, company_profile, requirements, benefits, salary_range)
  are concatenated into a single `model_text` column.
- TF-IDF (Term Frequency–Inverse Document Frequency) transforms the text into sparse numeric vectors.
- Categorical fields (employment_type, required_experience, etc.) are one-hot encoded.
- Numeric flags (telecommuting, has_company_logo, has_questions) are passed through.

**Model selection:**
Two models were trained and compared:
1. **Logistic Regression** — selected for deployment due to predict_proba support (calibrated scores)
   and interpretability. Threshold tuned to 0.78.
2. **Linear SVM** — higher balanced accuracy but no native probability output.

**Training environment:** Google Colab with scikit-learn 1.9.1, numpy 1.x, pandas < 3.0.

**Deployment environment:** Must match scikit-learn 1.9.1 to load the joblib artifact without
version mismatch warnings.
""")

st.divider()

# ─── Scoring formula ──────────────────────────────────────────────────────────
st.subheader("🧮 Scoring Formula & Thresholds")
st.markdown("""
**Composite score formula:**
```
composite_score = round(0.65 × model_risk_percent + 0.35 × rule_score)
```

| Component | Weight | Range | Source |
|---|---|---|---|
| `model_risk_percent` | 65% | 0–100 | Logistic Regression predict_proba × 100 |
| `rule_score` | 35% | 0–100 | Sum of triggered rule points (capped) |
| `composite_score` | — | 0–100 | Heuristic risk indicator |

**Risk categories:**
| Score | Category |
|---|---|
| 0–29 | 🟢 Lower detected risk — still verify independently |
| 30–59 | 🟡 Review recommended — investigate before proceeding |
| 60–100 | 🔴 High risk — verify before proceeding |

⚠️ **The composite score is a heuristic indicator, not a calibrated probability of fraud.**
""")

st.divider()

# ─── Rule engine ───────────────────────────────────────────────────────────────
st.subheader("📏 Rule Engine Checks")
st.markdown("""
| Rule | Severity | Points | What it checks |
|---|---|---|---|
| `missing_employer_info` | high/medium/low | 35/20/10 | Missing company name, profile, website, contact |
| `upfront_payment_language` | high | 35 | Keyword patterns for fees, deposits, advance payments |
| `unrealistic_earnings_claim` | medium | 20 | Guaranteed earnings, passive income, no-experience needed |
| `missing_deadline` | low | 10 | No application deadline |
| `expired_deadline` | medium | 20 | Deadline is in the past |
| `invalid_deadline` | low | 10 | Unparseable date |
| `invalid_email_format` | medium | 20 | Malformed email address |
| `free_email_domain` | low | 10 | Contact uses Gmail/Yahoo instead of company domain |
| `email_website_domain_mismatch` | medium | 20 | Email domain ≠ website domain |
| `missing_contact_details` | medium | 20 | No email and no phone |
""")

st.divider()

# ─── Limitations ───────────────────────────────────────────────────────────────
st.subheader("⚠️ Known Limitations")
config_path = ROOT / "artifacts" / "opportunityguard_config.json"
if config_path.exists():
    with config_path.open() as f:
        cfg = json.load(f)
    for lim in cfg.get("limitations", []):
        st.markdown(f"- {lim}")

st.markdown("""
- **Historical data:** The training dataset reflects fraud patterns up to its collection date.
  Novel scam tactics may not be captured.
- **False positives:** Legitimate listings may trigger warnings (e.g., legitimate training programs,
  legitimate international payment arrangements). A warning is not proof of fraud.
- **False negatives:** Sophisticated fraud listings may evade both the ML model and rule engine.
  Always verify employers independently.
- **No real-time verification:** The system cannot look up company registrations, website legitimacy,
  phone number validity, or social media presence.
- **Keyword matching limitations:** Rule patterns match text literally. Paraphrasing or obfuscation
  can evade them.
- **Calibration:** The composite score is a heuristic indicator. It is not a calibrated probability
  of fraud and should not be used as one.
- **SQLite persistence:** On Streamlit Community Cloud, the review database is stored in ephemeral
  deployment storage and will be lost on redeploy or restart. Use a persistent external database
  for production use.
""")

st.divider()

# ─── Deployment notes ──────────────────────────────────────────────────────────
st.subheader("🚀 Deployment")
st.markdown("""
**Local:**
```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\\Scripts\\activate      # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

**Streamlit Community Cloud:**
1. Push this repository to GitHub (ensure `artifacts/` is committed — see .gitignore notes).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Choose the repo, set the branch, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit will install `requirements.txt` automatically.

> ⚠️ The `artifacts/` directory (~38 MB duplicate index) must be committed to the repository.
> The `heldout_test_predictions.csv` is ~66 KB and is also committed for evaluation charts.
""")

st.divider()
st.caption("OpportunityGuard AI — Built for educational and demonstration purposes.")
