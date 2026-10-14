# 🛡️ OpportunityGuard AI

**ML-powered safety assessment for internship, job, and freelance listings.**  
Identify suspicious patterns, detect copied descriptions, calculate transparent risk scores,
and manage a human-review workflow — all from a local Streamlit application.

---

## Features

| Feature | Description |
|---|---|
| **ML Risk Score** | Logistic Regression pipeline (TF-IDF + one-hot) trained on 17,880 real job postings |
| **Rule Engine** | Deterministic checks: upfront payments, missing employer info, unrealistic claims, deadline validity, email/domain consistency |
| **Duplicate Detection** | Cosine similarity search against indexed reference descriptions |
| **Composite Score** | `round(0.65 × model_risk_percent + 0.35 × rule_score)` — a heuristic indicator, not a calibrated probability |
| **Human-Review Queue** | SQLite-backed case management with LEGITIMATE / SUSPICIOUS / INCONCLUSIVE decisions |
| **Assessment History** | Session-scoped history of recent assessments |
| **Exportable Reports** | Download assessment results as JSON or CSV |
| **Model Evaluation** | Real confusion matrix, score distribution, and radar chart from held-out test data |
| **Streamlit UI** | Dark-themed, multi-page, mobile-friendly interface |

---

## System Architecture

```
OpportunityGuard-AI/
├── app.py                         ← Dashboard (Streamlit entry point)
├── pages/
│   ├── 1_Assess_Listing.py        ← Full listing assessment form
│   ├── 2_Similar_Listings.py      ← Cosine similarity explorer
│   ├── 3_Review_Queue.py          ← Human-review SQLite workflow
│   ├── 4_Model_Evaluation.py      ← Real metrics and charts
│   └── 5_About.py                 ← Documentation and limitations
├── src/
│   ├── resources.py               ← Shared Streamlit cached loaders
│   ├── model_service.py           ← Pipeline loading and inference
│   ├── rule_engine.py             ← Deterministic rule checks
│   ├── duplicate_detector.py      ← TF-IDF similarity search
│   ├── risk_scoring.py            ← Composite score formula
│   ├── review_repository.py       ← SQLite CRUD for review cases
│   ├── validation.py              ← Input sanitisation
│   └── report_export.py           ← JSON/CSV export
├── artifacts/
│   ├── opportunityguard_model.joblib          ← Trained pipeline
│   ├── opportunityguard_duplicate_index.joblib ← Reference index (~37 MB)
│   ├── opportunityguard_config.json           ← Model config and thresholds
│   ├── final_test_metrics.csv                 ← Held-out test metrics
│   ├── model_comparison.csv                   ← LR vs SVM comparison
│   ├── heldout_test_predictions.csv           ← Raw predictions for charts
│   └── dataset_schema.json                    ← Dataset statistics
├── tests/                         ← pytest test suite
├── .streamlit/config.toml         ← Dark theme + server settings
├── requirements.txt               ← Pinned dependencies
└── .gitignore
```

---

## Dataset

- **Source:** [EMSCAD — Real or Fake Job Postings (Kaggle)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Records (after deduplication):** 17,880
- **Fraudulent:** 866 (~4.8%) | **Legitimate:** 17,014 (~95.2%)
- The model **does not** download the dataset at runtime.

---

## Model Training Summary

- **Algorithm:** Logistic Regression with scikit-learn Pipeline
- **Features:** TF-IDF on concatenated text fields + one-hot categorical + numeric flags
- **Decision threshold:** 0.78 (tuned to balance precision and recall)
- **Training environment:** Google Colab, scikit-learn 1.9.1, numpy 1.x, pandas < 3.0

### Held-Out Test Metrics

| Metric | Value |
|---|---|
| Accuracy | 98.55% |
| Balanced Accuracy | 87.92% |
| Fraud Precision | 92.52% |
| Fraud Recall | 76.15% |
| F1 Score | 83.54% |
| ROC-AUC | 0.9901 |
| PR-AUC | 0.9276 |
| MCC | 0.8322 |
| Brier Score | 0.0216 |
| Log Loss | 0.0882 |

> ⚠️ These metrics describe historical dataset performance. They are not predictions for individual listings.

---

## Local Setup

### Prerequisites

- Python 3.12 (recommended; tested with 3.12.7)
- pip

### Installation

```bash
# Clone or copy the project
git clone <your-repo-url>
cd OpportunityGuard-AI

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependencies (scikit-learn version MUST match training: 1.9.1)
pip install -r requirements.txt
```

### Launch the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Run the Test Suite

```bash
# Make sure the venv is activated
pytest tests/ -v
```

All tests use isolated temporary databases. The real `review_queue.db` is never modified by tests.

---

## Configuring Artifact Paths

All artifacts are loaded from `artifacts/` relative to the project root.  
If you need to change paths, edit the `ARTIFACTS_DIR` constants in:

- `src/model_service.py`
- `src/duplicate_detector.py`
- `src/review_repository.py`

Do **not** use absolute Colab paths (`/content/`) — they break on deployment.

---

## Risk Scoring Architecture & Interpretation

OpportunityGuard AI uses a transparent two-tier assessment combining a machine learning pipeline with deterministic heuristic rules:

```
composite_score = round(0.65 × model_risk_percent + 0.35 × rule_score)
```

### 1. Machine Learning Model Score
- **Classifier:** Logistic Regression pipeline with TF-IDF text vectorization and one-hot categorical features.
- **Decision Threshold:** **0.78** (tuned on held-out validation data to maximize precision and keep false positives minimal).
- **Interpretation:** The raw ML output is a statistical classification score reflecting vocabulary similarity to 17,880 historical job listings (EMSCAD). **It is not a calibrated real-world probability of fraud.** A listing is only flagged as an ML positive if `model_risk_percent >= 78%`.

### 2. Rule Engine Score (0–100)
- Deterministic heuristic checks detecting high-risk operational patterns:
  - **Upfront payment / registration fee demands:** +35 pts (negation-aware clause analysis)
  - **Unrealistic earnings / guaranteed income:** +25 pts
  - **Missing employer identity:** +15 pts
  - **Missing application deadline:** +10 pts
  - **Suspicious contact info / free email domain:** +15 pts

### 3. Composite Risk Bands
The final composite score determines the UI safety tier:

| Composite Score | Safety Tier | Meaning & Guidance |
|---|---|---|
| **0 – 29** | 🟢 **Lower Detected Risk** | No major heuristic red flags or ML anomalies detected. Standard precautions still apply. |
| **30 – 59** | 🟡 **Review Recommended** | Mild warning signs (e.g. missing employer identity or unverified contact details). Manual check recommended before applying. |
| **60 – 100** | 🔴 **High Risk** | Critical red flags detected (e.g., upfront payment demands, unrealistic income claims, or high ML anomaly score). Verify before proceeding. |

### 4. ML Decision vs. Composite Risk Category (Separate Decisions)
The ML binary flag and the composite risk category represent two separate, complementary evaluation layers:
- **ML Binary Flag:** Evaluates whether lexical and metadata patterns resemble known historical scam listings above the tuned decision threshold of **0.78** (78.0%).
- **Composite Risk Category:** Combines ML risk (65% weight) and heuristic rule score (35% weight) to place the listing into one of three risk tiers (0–29 Lower Risk, 30–59 Review Recommended, 60–100 High Risk).

> **Why are these separate decisions?**
> - An unflagged ML score (e.g., ML score = 45% < 78%) combined with a rule score of 70 yields:
>   $$\text{composite} = \text{round}(0.65 \times 45 + 0.35 \times 70) = \text{round}(29.25 + 24.50) = 54$$
>   This places the listing in the **Review Recommended** band (30–59), **not** the High Risk band.
> - To reach the **High Risk** band ($\ge 60$) with an unflagged ML score of 45%, the rule score must reach at least **88** (since $\text{round}(0.65 \times 45 + 0.35 \times 88) = \text{round}(29.25 + 30.80) = 60$).
> - Conversely, an ML-flagged listing (e.g., 80% $\ge 78%$) with 0 rule points yields $\text{round}(0.65 \times 80 + 0) = 52$, placing it in the **Review Recommended** composite band even though the ML model flagged it.

### 5. Human-Review Queue Routing Logic
A listing is routed to the human-review queue (`needs_human_review = True`) when **any** of the following conditions is satisfied:
1. **Model Flagged:** `model_flagged == True` (ML risk score $\ge 0.78$).
2. **High Rule Score:** `rule_score >= 35` (e.g., upfront payment demands or multiple severe warnings).
3. **Composite Risk Threshold:** `composite_score >= 30` (any listing categorized as Review Recommended or High Risk).

*OpportunityGuard AI provides heuristic risk indicators and review recommendations to assist students; it does not claim to prove fraud or guarantee authenticity. Independent due diligence is always recommended.*

---

## Deploying to Streamlit Community Cloud

1. **Push to GitHub:**  
   Commit all project files including the `artifacts/` folder:
   - `artifacts/opportunityguard_model.joblib`
   - `artifacts/opportunityguard_duplicate_index.joblib` (~37 MB)
   - `artifacts/opportunityguard_config.json`
   - `artifacts/final_test_metrics.csv`
   - `artifacts/model_comparison.csv`
   - `artifacts/heldout_test_predictions.csv`
   - `artifacts/dataset_schema.json`

   > 💡 If using Git LFS for large files:
   > ```bash
   > git lfs install
   > git lfs track "artifacts/*.joblib"
   > git add .gitattributes
   > git commit -m "Track large artifacts with Git LFS"
   > ```

2. **Deploy on Streamlit Community Cloud:**
   - Navigate to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Select your repository and branch.
   - Set **Main file path** to: `app.py`
   - Click **Deploy**. Streamlit automatically builds the environment from `requirements.txt`.

### Database Persistence & Cloud Storage Limitations

> ⚠️ **Important SQLite Persistence Note:**  
> Streamlit Community Cloud uses an ephemeral container filesystem. Any local file written at runtime—including the default `review_queue.db` SQLite database—**will be reset when the container sleeps, restarts, or is redeployed.**

#### Making SQLite Path Configurable
OpportunityGuard AI supports custom database paths through environment variables or Streamlit secrets:

1. **Via Environment Variable:**
   ```bash
   # Linux / macOS
   export OPPORTUNITYGUARD_DB_PATH="/path/to/persistent/review_queue.db"

   # Windows PowerShell
   $env:OPPORTUNITYGUARD_DB_PATH="C:\path\to\review_queue.db"
   ```

2. **Via Streamlit Secrets (`.streamlit/secrets.toml`):**
   ```toml
   [database]
   path = "/mount/data/review_queue.db"
   ```

#### Optional: Connecting an External Managed Database (Supabase / PostgreSQL)
For persistent production deployments, connect a hosted PostgreSQL database (such as Supabase, Neon, or Railway):
1. Create a free PostgreSQL instance on Supabase.
2. Add your connection string to `.streamlit/secrets.toml` or Streamlit Cloud Secrets:
   ```toml
   [postgres]
   url = "postgresql://postgres:password@db.supabase.co:5432/postgres"
   ```
3. In `src/review_repository.py`, create a PostgreSQL adapter adhering to the same interface (`insert_review_case`, `fetch_review_cases`, `update_review_decision`).

---

## Automated Testing

OpportunityGuard AI includes a comprehensive regression test suite using `pytest`:

```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
source .venv/bin/activate # Linux / macOS

# Run all automated tests
pytest tests/ -v
```

The test suite covers 161 tests across 7 test suites:
- **Negation-aware upfront payment detection:** 10 negative disclaimer patterns, 6 positive fee demands, multi-clause contrastive sentences ("We do not charge an application fee, but you must pay ₹999 to register"), and edge cases.
- **Unrealistic earnings & salary checks:** Distinguishes normal salary ranges, negotiable compensation, and explicit disclaimers ("does not guarantee earnings") from fraudulent work-from-home claims.
- **Risk scoring & boundary tests:** Threshold transitions (0, 29, 30, 59, 60, 100), banker's rounding compliance, and clamping.
- **Review queue operations:** Database initialization without data loss, single-click submission, duplicate case prevention, status filtering, and decision updates.
- **Model evaluation & reports:** Robust handling of metrics, SVM decision-function boundaries, and JSON/CSV export generation.

---

## Known Limitations and Troubleshooting

1. **scikit-learn Version Pinning:**  
   The model pipeline was trained using scikit-learn **1.9.1**. Using older or newer minor versions can trigger `InconsistentVersionWarning` or prediction discrepancies. Ensure `scikit-learn==1.9.1` is installed.

2. **PyArrow & NumPy Compatibility:**  
   The project pins `numpy==1.26.4` and `pyarrow==18.1.0`. Newer PyArrow builds (e.g. 19+) enforce NumPy >= 2.0; the pinned versions ensure seamless DataFrame rendering on the Model Evaluation page without import conflicts.

3. **Linear SVM Threshold Explanation:**  
   The Model Evaluation page compares Logistic Regression against Linear SVM. Linear SVM's threshold is displayed as **0.0000** because it represents the hyperplane margin boundary of `decision_function()`, not a probability. Logistic Regression outputs calibrated likelihoods evaluated at threshold **0.7800**.

4. **Historical Dataset Scope:**  
   The model is trained on the EMSCAD benchmark dataset (17,880 listings). While highly effective at identifying recurring scam tropes and vocabulary anomalies, scammers evolve new tactics. Always apply the safety disclaimer and verify employers directly.

---

## Disclaimers & Safety

- **A flagged listing is not proven fraud.** False positives occur.
- **A clean score does not guarantee legitimacy.** False negatives occur.
- OpportunityGuard AI is an assistive screening tool designed for student guidance and safety education.
- Always verify employer registrations through official government and corporate registries before sharing personal information or signing contracts.

---

*Built for educational and safety demonstration purposes. OpportunityGuard AI is not a substitute for independent verification.*
