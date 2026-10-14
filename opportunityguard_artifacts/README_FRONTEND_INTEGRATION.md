# OpportunityGuard AI — Antigravity Integration

Artifacts: fitted model, duplicate index, config, metrics and sample API request/response.

Suggested stack: React + Vite frontend, FastAPI backend, SQLite review queue. Suggested endpoints: GET /api/health, POST /api/assess, GET /api/reviews, PATCH /api/reviews/{id}. Load the model with joblib.load and build a one-row DataFrame using config['model_fields']. Keep the fitted preprocessing pipeline; do not refit TF-IDF or encoders in the frontend. Display model score, rule score and composite score separately. The composite is not a calibrated probability. A model prediction is not proof of fraud.
