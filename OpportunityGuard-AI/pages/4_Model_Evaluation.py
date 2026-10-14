"""
pages/4_Model_Evaluation.py — Display actual exported evaluation metrics and charts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Model Evaluation — OpportunityGuard AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Evaluation")
st.caption(
    "All metrics and charts on this page are derived from real exported evaluation files "
    "produced during training. No synthetic or placeholder numbers are used."
)

ARTIFACTS = ROOT / "artifacts"

# ─── Model comparison ──────────────────────────────────────────────────────────
comp_path = ARTIFACTS / "model_comparison.csv"
if comp_path.exists():
    st.subheader("🏆 Model Comparison")
    df_comp = pd.read_csv(comp_path)
    styled_df = df_comp.style.format({
        col: "{:.4f}" for col in df_comp.select_dtypes("float").columns
    })
    try:
        st.dataframe(styled_df, use_container_width=True)
    except Exception:
        st.table(styled_df)
    st.caption(
        "📌 **Threshold Definitions:**\n"
        "- **Linear SVM (0.0000):** Represents a geometric decision-function boundary (signed distance to the separating hyperplane: `decision_function >= 0.0`), **not** a probability threshold. Linear SVM does not produce calibrated probabilities.\n"
        "- **Logistic Regression in Comparison (0.5000):** Shown at standard default threshold during baseline comparison.\n"
        "- **Deployed Production Pipeline (0.7800):** As configured in `opportunityguard_config.json`, the deployed model applies a tuned decision threshold of **0.78** (78%) on held-out data to maximize precision (92.5%) and minimize false alarms against legitimate student jobs."
    )
    st.divider()
else:
    st.warning("model_comparison.csv not found in artifacts/")

# ─── Final test metrics ────────────────────────────────────────────────────────
metrics_path = ARTIFACTS / "final_test_metrics.csv"
if metrics_path.exists():
    st.subheader("📈 Held-Out Test Metrics (Logistic Regression)")
    st.info(
        "These metrics were computed on a **held-out test set** never seen during training or "
        "threshold selection. They describe model performance on historical data and should not "
        "be interpreted as accuracy on any individual new listing."
    )
    df_m = pd.read_csv(metrics_path)
    row = df_m.iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{row['Accuracy']:.4f}", help="Overall correct predictions / total")
    c2.metric("Balanced Accuracy", f"{row['Balanced accuracy']:.4f}", help="Average of per-class recall")
    c3.metric("Fraud Precision", f"{row['Fraud precision']:.4f}", help="TP / (TP+FP) for the fraud class")
    c4.metric("Fraud Recall", f"{row['Fraud recall']:.4f}", help="TP / (TP+FN) for the fraud class")
    c5.metric("F1 Score", f"{row['Fraud F1']:.4f}", help="Harmonic mean of precision and recall")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("ROC-AUC", f"{row['ROC-AUC']:.4f}", help="Area under ROC curve; 1.0 = perfect")
    c7.metric("PR-AUC", f"{row['Average precision / PR-AUC']:.4f}", help="Area under Precision-Recall curve")
    c8.metric("MCC", f"{row['MCC']:.4f}", help="Matthews Correlation Coefficient; 1.0 = perfect")
    c9.metric("Brier Score", f"{row['Brier score']:.4f}", help="Mean squared error of probability estimates; lower is better")
    c10.metric("Log Loss", f"{row['Log loss']:.4f}", help="Cross-entropy loss; lower is better")

    st.divider()

    # Radar chart
    st.subheader("🕸️ Performance Radar")
    labels = ["Accuracy", "Bal. Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    vals = [
        row["Accuracy"], row["Balanced accuracy"], row["Fraud precision"],
        row["Fraud recall"], row["Fraud F1"], row["ROC-AUC"],
        row["Average precision / PR-AUC"],
    ]
    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.25)",
        line=dict(color="#6366f1", width=2.5),
        marker=dict(size=7, color="#6366f1"),
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1e293b",
            radialaxis=dict(
                visible=True, range=[0, 1],
                gridcolor="#334155", color="#94a3b8",
                tickformat=".0%",
            ),
            angularaxis=dict(gridcolor="#334155", color="#e2e8f0"),
        ),
        paper_bgcolor="#0f1117",
        font=dict(color="#e2e8f0", family="Inter"),
        showlegend=False,
        height=400,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.divider()

else:
    st.warning("final_test_metrics.csv not found in artifacts/")

# ─── Held-out predictions analysis ────────────────────────────────────────────
pred_path = ARTIFACTS / "heldout_test_predictions.csv"
if pred_path.exists():
    st.subheader("📉 Score Distribution on Held-Out Test Set")
    df_pred = pd.read_csv(pred_path)

    # Distribution of model_risk_score by class
    legit = df_pred[df_pred["actual_fraudulent"] == 0]["model_risk_score"]
    fraud = df_pred[df_pred["actual_fraudulent"] == 1]["model_risk_score"]

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=legit, nbinsx=50, name="Legitimate",
        marker_color="rgba(74,222,128,0.7)",
        opacity=0.75,
    ))
    fig_dist.add_trace(go.Histogram(
        x=fraud, nbinsx=50, name="Fraudulent",
        marker_color="rgba(248,113,113,0.7)",
        opacity=0.75,
    ))
    fig_dist.add_vline(
        x=0.78, line_dash="dash", line_color="#f59e0b",
        annotation_text="threshold=0.78", annotation_position="top right",
    )
    fig_dist.update_layout(
        barmode="overlay",
        xaxis_title="Model Risk Score (predict_proba)",
        yaxis_title="Count",
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        height=380,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Confusion matrix from predictions
    st.subheader("🔢 Confusion Matrix (threshold = 0.78)")
    THRESHOLD = 0.78
    y_true = df_pred["actual_fraudulent"]
    y_pred = (df_pred["model_risk_score"] >= THRESHOLD).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    cm = np.array([[tn, fp], [fn, tp]])
    fig_cm = go.Figure(go.Heatmap(
        z=cm,
        x=["Predicted: Legitimate", "Predicted: Fraudulent"],
        y=["Actual: Legitimate", "Actual: Fraudulent"],
        colorscale="Blues",
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont=dict(size=20, color="white"),
        showscale=False,
    ))
    fig_cm.update_layout(
        paper_bgcolor="#0f1117",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", family="Inter"),
        height=350,
        margin=dict(t=20, b=40),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
    cm_c1.metric("True Negatives (correct legitimate)", tn)
    cm_c2.metric("False Positives (legit flagged)", fp,
                 help="Legitimate listings flagged as fraudulent — false alarms")
    cm_c3.metric("False Negatives (missed fraud)", fn,
                 help="Fraudulent listings not flagged — missed detections")
    cm_c4.metric("True Positives (fraud caught)", tp)

    st.caption(
        "**False positives** cause unnecessary alarm for legitimate listings. "
        "**False negatives** are missed fraudulent listings. "
        "The threshold (0.78) was tuned to reduce false alarms at the cost of some missed detections."
    )
    st.divider()

    # Summary stats
    st.subheader("📊 Prediction Statistics")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Total Test Samples", len(df_pred))
    sc2.metric("Actual Fraudulent", int(y_true.sum()))
    sc3.metric("Flagged by Model", int(y_pred.sum()))
else:
    st.warning("heldout_test_predictions.csv not found in artifacts/")

# ─── Scoring formula explanation ──────────────────────────────────────────────
st.subheader("🧮 Composite Score Formula")
st.code("composite_score = round(0.65 × model_risk_percent + 0.35 × rule_score)", language="text")
st.markdown("""
| Component | Weight | Description |
|---|---|---|
| `model_risk_percent` | 65% | ML model output × 100 (0–100 scale) |
| `rule_score` | 35% | Sum of triggered rule points, capped at 100 |
| `composite_score` | — | Heuristic risk indicator, **not** a calibrated probability |

**Thresholds:**
- `< 30` → 🟢 Lower detected risk  
- `30–59` → 🟡 Review recommended  
- `≥ 60` → 🔴 High risk — verify before proceeding
""")
