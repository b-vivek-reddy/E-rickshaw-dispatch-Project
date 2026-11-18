# 🛺 AI-Based Campus E-Rickshaw Demand & Intelligent Dispatch System

**INT395: Supervised Learning — Course Project (Project 04)**
Team size: 3 · Project duration: 2–2.5 months

An end-to-end ML system that predicts campus e-rickshaw demand, driver availability,
and ride wait times, and uses those predictions to drive an intelligent dispatch
policy with dynamic search-radius expansion — deployed as an interactive Streamlit
app with Student, Driver, and Admin views.

---

## 1. Problem

Campus e-rickshaws face unpredictable demand, long passenger waits, and drivers that
are unevenly distributed, leaving some ride requests unfulfilled. This project builds
four ML models and a decision layer that combines them into a working dispatch system,
rather than stopping at a single trained model in a notebook.

| # | Model | Type | Target |
|---|---|---|---|
| 1 | Demand Prediction | Regression | Ride requests per zone per hour |
| 2 | Driver Availability Prediction | Classification | Will a driver be available in the next 15-min slot? |
| 3 | ETA / Wait-Time Prediction | Regression | Minutes until a matched driver arrives |
| 4 | Failed-Request Prediction | Classification | Will this request fail to find any driver? |

Model 4's output directly drives the **dynamic search-radius** decision in the
dispatch engine (`src/dispatch.py`) — this is the "explainability → decision" link
the rubric asks for, not just a SHAP plot sitting next to an unrelated model.

## 2. Results summary (held-out test set, chronological split)

| Model | Metric | Baseline | Best Model | Best Model Score |
|---|---|---|---|---|
| Demand Prediction | MAE | 0.44 (mean) | XGBoost | **0.35** |
| Driver Availability | ROC-AUC | 0.50 (majority) | XGBoost (Tuned) | **0.63** |
| ETA Prediction | R² | -0.00 (mean) | Voting Ensemble | **0.75** |
| Failed-Request Prediction | F1 / ROC-AUC | 0.00 / 0.50 | XGBoost (Tuned) | **0.62 / 0.96** |

**Dispatch policy comparison** (fulfilment rate, test period): static-radius baseline
**92.2%** → ML-informed dynamic-radius policy **93.6%**.

Full metrics, validation comparisons across all candidate algorithms, and discussion
of *why* each score looks the way it does (sparse counts for demand, stochastic
driver behaviour for availability) are in `notebooks/01_EDA_and_Modeling.ipynb` and
`models/*_metrics.json`.

## 3. Repository structure

```
erickshaw-dispatch/
├── data/
│   ├── generate_synthetic_data.py   # documented, seeded simulation (see data/README.md)
│   ├── README.md                    # dataset source, characteristics, limitations
│   ├── raw/                         # generated CSVs (locations, drivers, rides, shifts)
│   └── processed/                   # engineered modeling datasets
├── notebooks/
│   ├── build_notebook.py            # builds + executes the notebook below
│   └── 01_EDA_and_Modeling.ipynb    # EDA, model comparison, SHAP, dispatch evaluation
├── src/
│   ├── utils.py                     # shared constants, distance/time helpers
│   ├── feature_engineering.py       # builds the 3 modeling datasets (leakage-safe)
│   ├── model_utils.py               # shared metrics/plotting helpers
│   ├── train_demand.py              # Model 1
│   ├── train_availability.py        # Model 2
│   ├── train_eta_failed.py          # Models 3 & 4
│   └── dispatch.py                  # intelligent matching + dynamic radius + route analysis
├── app/
│   └── streamlit_app.py             # DEPLOYMENT — Student / Driver / Admin views
├── models/                          # saved *.joblib pipelines + *_metrics.json
├── reports/figures/                 # model comparison, residuals, SHAP, confusion matrices
├── tests/test_pipeline.py           # pytest smoke tests for the trained pipeline
├── requirements.txt
└── README.md
```

## 4. Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Reproducing the pipeline from scratch

```bash
# 1. Generate the (seeded, reproducible) synthetic dataset
python data/generate_synthetic_data.py

# 2. Build the leakage-safe feature-engineered datasets
cd src
python feature_engineering.py

# 3. Train all four models (each script prints validation comparisons + saves artifacts)
python train_demand.py
python train_availability.py
python train_eta_failed.py        # trains both ETA and Failed-Request models
cd ..

# 4. (Optional) Rebuild the executed EDA/modeling notebook
cd notebooks && python build_notebook.py && cd ..

# 5. Run the test suite
pytest tests/ -v
```

## 6. Running the deployed app

```bash
streamlit run app/streamlit_app.py
```

Then open the printed local URL. Use the sidebar to switch between **Student**
(request a ride, see predicted wait time), **Driver** (see nearby predicted demand
to position yourself), and **Admin** (demand heatmap, worst-served routes, model
performance, SHAP explanations) — all backed by the same trained models in `models/`.

## 7. Design notes / justification for technical choices

- **Why synthetic data?** No public dataset exists for campus e-rickshaw demand; see
  `data/README.md` for the full, transparent documentation of every generation
  assumption (this is disclosed, not hidden, per the course's academic-integrity
  policy).
- **Why XGBoost + ensembling over a single model?** Each task compares a baseline,
  linear/logistic regression, Random Forest, and XGBoost, then tunes XGBoost via
  `RandomizedSearchCV` with `TimeSeriesSplit`, and finally tries a Voting/Stacking
  ensemble — the better of the tuned single model or the ensemble is kept, so the
  choice is evidence-based per task rather than assumed.
- **Why a chronological split, not random k-fold?** This is time-ordered operational
  data; a random split would let the model "see the future" during training.
- **Why SHAP feeds into the dispatch design?** The Failed-Request model's SHAP
  summary shows `nearby_available_drivers` and `rolling_zone_fail_rate` dominate
  failure risk — `dispatch.py` uses exactly that model's output to decide how wide to
  start the driver search, so explainability informed an actual system decision.
- **Why one Streamlit app with three views instead of three apps?** Functionally
  equivalent role-based access with a fraction of the deployment and maintenance
  surface area.

## 8. Rubric component mapping

| Rubric component | Where it's satisfied |
|---|---|
| Problem Definition | README §1, notebook §1 |
| Dataset & Data Acquisition | `data/README.md`, `data/generate_synthetic_data.py` |
| EDA | `notebooks/01_EDA_and_Modeling.ipynb` §3 |
| Preprocessing & Feature Engineering | `src/feature_engineering.py`, notebook §4 |
| Model Development (multiple models) | `src/train_*.py` — 4 tasks × 4+ algorithms each |
| Model Evaluation (not accuracy-only) | MAE/RMSE/R² for regression; Precision/Recall/F1/ROC-AUC for classification |
| Ensemble Learning / Hyperparameter Optimization | `RandomizedSearchCV` + Voting/Stacking in every `train_*.py` |
| Explainable AI | SHAP in every `train_*.py`, notebook §8, Admin app tab |
| ML Pipeline | `sklearn.Pipeline` + `ColumnTransformer` in every `train_*.py` |
| Deployment | `app/streamlit_app.py` |
| GitHub Repository | this repository structure |

## 9. Known limitations

See `data/README.md` §Limitations and notebook §10 for a full discussion (synthetic
data, sparse-count demand ceiling, stochastic availability, Euclidean vs. real-path
distance). These are stated explicitly rather than glossed over, per the rubric's
emphasis on honest methodology over polished accuracy numbers.

## 10. Team contribution template

*(Fill in for submission — every member must be able to explain the full system at
viva per the course's academic-integrity policy.)*

| Area | Owner |
|---|---|
| Data simulation & documentation | |
| EDA & preprocessing/feature engineering | |
| Model development (Demand, Availability) | |
| Model development (ETA, Failed-Request), ensembling & tuning | |
| Explainable AI (SHAP) | |
| Dispatch engine & route analysis | |
| Streamlit deployment | |
| Testing, documentation, README | |
