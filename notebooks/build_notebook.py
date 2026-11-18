"""Builds notebooks/01_EDA_and_Modeling.ipynb programmatically and executes it so the
submitted notebook contains real, reproducible outputs rather than empty cells."""
import nbformat as nbf
from nbclient import NotebookClient
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ----------------------------------------------------------------------------
md("""# AI-Based Campus E-Rickshaw Demand & Intelligent Dispatch System
### INT395 — Supervised Learning | Course Project (Project 04)

**Pipeline covered in this notebook:** Problem → Data → EDA → Preprocessing → Feature
Engineering → Model Development → Model Comparison → Optimization → Explainability →
Validation.

Production code (reusable, tested) lives in `src/`; this notebook tells the *analysis
story* and references the same modules so there is a single source of truth — it does
not duplicate the modeling logic.
""")

# ----------------------------------------------------------------------------
md("## 1. Problem Definition\n\n"
   "**Real-world problem:** Campus e-rickshaws suffer from unpredictable ride demand, "
   "long passenger waiting times, and drivers that are unevenly distributed across the "
   "campus, causing some requests to go completely unfulfilled.\n\n"
   "**Target users/stakeholders:** Students/staff requesting rides, e-rickshaw drivers, "
   "and campus transportation administrators.\n\n"
   "**ML objective:** (1) predict ride demand ahead of time, (2) predict whether a given "
   "driver will be available, (3) predict the wait time a rider should expect, and "
   "(4) predict whether a request is at risk of going unfulfilled so the dispatcher can "
   "react before it happens.\n\n"
   "**Input variables:** time of day/week, weather, exam-period flag, pickup/dropoff "
   "zone characteristics, distance, nearby driver supply, historical zone performance.\n\n"
   "**Expected output:** demand forecasts, availability probabilities, ETA estimates, "
   "and failure-risk scores feeding an intelligent dispatch policy.\n\n"
   "**Success criteria:** meaningfully beat a naive baseline on each task (MAE/F1/AUC), "
   "and show the ML-informed dispatch policy improves fulfilment rate over the static "
   "rule-based policy.")

# ----------------------------------------------------------------------------
md("## 2. Dataset & Data Acquisition\n\n"
   "See `data/README.md` for full documentation. In short: no public dataset exists for "
   "this problem, so a parameterized simulation (`data/generate_synthetic_data.py`) "
   "produces realistic campus mobility data with a fixed random seed, modeling class/"
   "meal-time demand peaks, weekday/exam/weather effects, and **deliberately scarce "
   "driver supply** (16 drivers across 15 zones) so the waiting/failure problem is real, "
   "not trivial.")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 110

ROOT = Path('..').resolve()
rides = pd.read_csv(ROOT / 'data/raw/ride_requests.csv', parse_dates=['timestamp','date'])
shifts = pd.read_csv(ROOT / 'data/raw/driver_shift_log.csv', parse_dates=['date'])
drivers = pd.read_csv(ROOT / 'data/raw/drivers.csv')
locations = pd.read_csv(ROOT / 'data/raw/locations.csv')

print('Ride requests:', rides.shape)
print('Driver shift log:', shifts.shape)
print('Drivers:', drivers.shape)
print('Locations:', locations.shape)
rides.head()""")

code("""rides.info()
print()
print('Missing values per column:')
print(rides.isna().sum()[rides.isna().sum() > 0])""")

md("**Note on missing values:** `wait_time_minutes` and `assigned_driver_id` are "
   "missing *by construction* for failed requests (no driver was ever matched, so no "
   "wait time was ever observed) — this is not a data-quality problem and must not be "
   "mean-imputed, since that would hide the failure signal we want to model.")

# ----------------------------------------------------------------------------
md("## 3. Exploratory Data Analysis")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
rides.groupby('hour').size().plot(kind='bar', ax=axes[0], color='#4C72B0')
axes[0].set_title('Ride Requests by Hour of Day')
axes[0].set_xlabel('Hour'); axes[0].set_ylabel('Request Count')

order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
wd = rides.groupby('weekday').size()
wd.index = [order[i] for i in wd.index]
wd.reindex(order).plot(kind='bar', ax=axes[1], color='#DD8452')
axes[1].set_title('Ride Requests by Day of Week')
plt.tight_layout(); plt.show()""")

code("""pivot = rides.pivot_table(index='pickup_zone', columns='hour', values='request_id', aggfunc='count', fill_value=0)
plt.figure(figsize=(14, 6))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3)
plt.title('Demand Heatmap: Pickup Zone x Hour of Day')
plt.tight_layout(); plt.show()""")

code("""fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, flag, label in zip(axes, ['is_weekend','is_exam_period','is_raining'],
                             ['Weekend', 'Exam Period', 'Raining']):
    means = rides.groupby(flag).size() / rides[flag].nunique()
    rides.groupby(['date', flag]).size().reset_index(name='n').groupby(flag)['n'].mean().plot(
        kind='bar', ax=ax, color=['#55A868','#C44E52'])
    ax.set_title(f'Avg Daily Requests: {label}')
    ax.set_xticklabels(['No','Yes'], rotation=0)
plt.tight_layout(); plt.show()""")

md("**Observed pattern:** demand is higher on weekdays (classes in session), spikes "
   "during exam weeks (library/academic corridor), and rises noticeably on rainy days — "
   "confirming the simulation's documented assumptions actually show up in the data, "
   "the way a real deployment's EDA would need to confirm against operational logs.")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
sns.histplot(rides['distance_km'], bins=30, ax=axes[0], color='#4C72B0', kde=True)
axes[0].set_title('Trip Distance Distribution')

fulfilled = rides[~rides['request_failed']]
sns.histplot(fulfilled['wait_time_minutes'], bins=30, ax=axes[1], color='#55A868', kde=True)
axes[1].set_title('Wait Time Distribution (Fulfilled Requests)')
plt.tight_layout(); plt.show()

print('Failed-request rate:', round(rides['request_failed'].mean()*100, 2), '%  (class imbalance to handle in Model 4)')
print('Wait time (fulfilled) — median: %.1f min, 90th pct: %.1f min' % (
    fulfilled['wait_time_minutes'].median(), fulfilled['wait_time_minutes'].quantile(0.9)))""")

code("""fail_by_zone = rides.groupby('pickup_zone')['request_failed'].mean().sort_values(ascending=False)
plt.figure(figsize=(9,5))
fail_by_zone.plot(kind='barh', color='#C44E52')
plt.title('Failed-Request Rate by Pickup Zone'); plt.xlabel('Failure Rate')
plt.tight_layout(); plt.show()""")

code("""numeric_cols = ['hour','weekday','distance_km','wait_time_minutes','search_radius_expansions']
corr = rides[numeric_cols + ['request_failed']].astype(float).corr()
plt.figure(figsize=(6.5,5.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix (numeric features)')
plt.tight_layout(); plt.show()""")

code("""avail_rate = shifts[shifts['status'] != 'off_shift'].assign(
    is_avail=lambda d: (d['status']=='available').astype(int)
).groupby('hour_slot')['is_avail'].mean()
plt.figure(figsize=(11,4.3))
avail_rate.plot(color='#4C72B0')
plt.title('Driver Availability Rate Across the Day (on-shift drivers only)')
plt.xlabel('Hour of day (6 = 6 AM, 29.75 ≈ 5:45 AM next day)'); plt.ylabel('P(available | on shift)')
plt.tight_layout(); plt.show()""")

md("**EDA summary / patterns found:** (1) demand peaks at class-change and meal times; "
   "(2) driver availability *drops* during the same peak windows because more drivers "
   "are mid-trip — this supply/demand mismatch is the direct cause of waiting time and "
   "failed requests; (3) a handful of peripheral zones (hostels, back gate) show "
   "disproportionately high failure rates, which is investigated further in the "
   "Route/Failed-Request Analysis section below; (4) the failed-request target is "
   "imbalanced (~8%), requiring class-weighting during model development.")

# ----------------------------------------------------------------------------
md("## 4. Data Preprocessing & Feature Engineering\n\n"
   "Implemented in `src/feature_engineering.py`. Summary of engineering applied:\n\n"
   "- **Cyclical encoding** of hour-of-day and day-of-week (sin/cos) so the model sees "
   "midnight and 11 PM as close together, not numerically far apart.\n"
   "- **Lag/rolling features** (previous-hour demand, 7-day rolling average demand per "
   "zone-hour, rolling zone-level failure rate and wait time) — all shifted so no "
   "feature ever uses information from the future relative to the row it describes.\n"
   "- **Spatial feature**: `nearby_available_drivers`, computed by multiplying a "
   "per-zone available-driver-count vector by a zone-adjacency matrix (1 if within "
   "1.5 km), giving a live 'local supply' signal for each request.\n"
   "- **One-hot encoding** for categorical zone types / shift types via "
   "`sklearn.compose.ColumnTransformer`, **standard scaling** for numeric features — "
   "both wrapped inside an `sklearn.Pipeline` (Mandatory Component 9: ML Pipeline) so "
   "preprocessing is never fit on validation/test data.\n"
   "- **Chronological train/validation/test split** (70/15/15 by simulated day) instead "
   "of a random row split, since this is time-ordered operational data — a random split "
   "would leak future information into training.")

code("""demand_df = pd.read_csv(ROOT / 'data/processed/demand_dataset.csv')
eta_df = pd.read_csv(ROOT / 'data/processed/eta_failed_dataset.csv')
avail_df = pd.read_csv(ROOT / 'data/processed/availability_dataset.csv')
print('Demand modeling dataset:', demand_df.shape)
print('ETA/Failed-request modeling dataset:', eta_df.shape)
print('Driver availability modeling dataset:', avail_df.shape)
eta_df[['hour_sin','hour_cos','distance_km','nearby_available_drivers',
        'rolling_zone_wait_avg','rolling_zone_fail_rate','request_failed']].head()""")

# ----------------------------------------------------------------------------
md("## 5–7. Model Development, Comparison, Ensembling & Hyperparameter Optimization\n\n"
   "Full training code: `src/train_demand.py`, `src/train_availability.py`, "
   "`src/train_eta_failed.py`. For every task we compare a **naive baseline** against "
   "**Linear/Logistic Regression**, **Random Forest**, **XGBoost**, a "
   "**RandomizedSearchCV-tuned XGBoost** (time-aware CV via `TimeSeriesSplit`), and a "
   "**Voting/Stacking ensemble**. The best model by validation performance is refit on "
   "train+validation and evaluated exactly once on the held-out test set below.")

code("""def show_metrics(name, path):
    with open(ROOT / 'models' / path) as f:
        m = json.load(f)
    print(f'=== {name} ===')
    print('Best model:', m['best_model'])
    print('Validation results:')
    display(pd.DataFrame(m['validation_results']).T)
    print('Final TEST metrics:', m['test_metrics'])
    print()

from IPython.display import display
show_metrics('Model 1: Demand Prediction (regression)', 'demand_model_metrics.json')""")

code("show_metrics('Model 2: Driver Availability Prediction (classification)', 'availability_model_metrics.json')")

code("show_metrics('Model 3: ETA / Wait-Time Prediction (regression)', 'eta_model_metrics.json')")

code("show_metrics('Model 4: Failed-Request Prediction (classification)', 'failed_model_metrics.json')")

md("### Reading the results\n\n"
   "- **Demand Prediction** (MAE≈0.35, R²≈0.20 at zone-hour granularity): modest R² is "
   "expected — predicting an individual zone's request count in a single hour is a "
   "sparse, near-Poisson count problem with a lot of irreducible randomness; the model "
   "still beats the mean baseline by ~20% on MAE and is far more useful aggregated to "
   "zone/time-block level for planning, which the Admin Dashboard does.\n"
   "- **Driver Availability** (AUC≈0.63–0.66): availability has a genuinely stochastic "
   "component in how drivers accept/decline rides in the simulation (and in real life — "
   "breaks, detours, personal calls); the model still meaningfully beats a 50/50 "
   "baseline and the most informative signal is *recent own status* "
   "(`prev_slot_available`), which is intuitive.\n"
   "- **ETA Prediction** (R²≈0.75): strong — nearby driver supply and historical "
   "zone-level wait time are highly predictive, as expected.\n"
   "- **Failed-Request Prediction** (AUC≈0.96, F1≈0.62): very strong — local driver "
   "scarcity (`nearby_available_drivers`) and the rolling zone failure rate dominate, "
   "which makes this model genuinely useful for the dynamic-radius dispatch decision "
   "(see `src/dispatch.py`).\n\n"
   "**Accuracy alone was not used as the deciding metric anywhere** — MAE/RMSE/R² "
   "for regression tasks, and Precision/Recall/F1/ROC-AUC (not Accuracy) for the "
   "imbalanced Failed-Request classifier, per the rubric's explicit instruction.")

# ----------------------------------------------------------------------------
md("## 8. Explainable AI\n\n"
   "SHAP (`TreeExplainer`) is applied to the tuned XGBoost model for every task. "
   "Summary plots are generated in the training scripts and saved to "
   "`reports/figures/*_shap_summary.png`. Interpretation (not just plots):\n\n"
   "- **Demand**: `rolling7_avg_zone_hour` and `lag_24h_same_zone_hour` dominate — "
   "recent history at the same zone/hour is the strongest predictor of near-term "
   "demand, consistent with the periodic class/meal-time structure found in EDA.\n"
   "- **ETA**: `nearby_available_drivers` has the largest, most negative SHAP impact — "
   "more local supply directly shortens predicted wait, exactly as the dispatch logic "
   "assumes.\n"
   "- **Failed-Request**: `nearby_available_drivers` and `rolling_zone_fail_rate` are "
   "the dominant drivers of failure risk, which is *why* `src/dispatch.py` uses this "
   "model's output to decide the dispatch search radius — the explanation directly "
   "informed a design decision, not just a plot.")

code("""from PIL import Image
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
imgs = ['eta_shap_summary.png', 'failed_shap_summary.png',
        'demand_shap_summary.png', 'availability_shap_summary.png']
titles = ['ETA Prediction', 'Failed-Request Prediction', 'Demand Prediction', 'Driver Availability']
for ax, img, title in zip(axes.flat, imgs, titles):
    im = Image.open(ROOT / 'reports/figures' / img)
    ax.imshow(im); ax.axis('off'); ax.set_title(title)
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md("## 9. Dispatch Optimization — Route / Failed-Request Analysis\n\n"
   "`src/dispatch.py` combines the trained Failed-Request and ETA models into an actual "
   "dispatch **policy**: the search radius now starts wider for requests the model "
   "already flags as high-risk, instead of always starting narrow and reacting after "
   "the first failed search. Evaluated on the held-out test period:")

code("""import sys
sys.path.insert(0, str(ROOT / 'src'))
from dispatch import evaluate_policy
baseline, ml_informed = evaluate_policy()
pd.DataFrame({'Policy': ['Static Radius (baseline)', 'ML-Informed Dynamic Radius'],
              'Fulfilment Rate': [baseline, ml_informed]})""")

code("""from dispatch import DispatchEngine
engine = DispatchEngine()
top_poor_routes = engine.route_analysis(rides, top_n=10)
top_poor_routes""")

md("**Route Analysis finding:** the worst-served routes are concentrated on trips to/"
   "from the peripheral hostel zones and Back Gate — these are the routes campus "
   "transportation administrators should prioritize for additional driver coverage, "
   "which is surfaced directly on the Admin Dashboard in the deployed app.")

# ----------------------------------------------------------------------------
md("## 10. Validation, Limitations & Conclusion\n\n"
   "**Validation strategy:** chronological 70/15/15 split; hyperparameter tuning used "
   "only the validation set (via `TimeSeriesSplit` cross-validation on the training "
   "portion); every test metric above was computed exactly once after model selection, "
   "with no further tuning — avoiding test-set leakage.\n\n"
   "**Limitations:**\n"
   "1. Data is simulated, not observed (see `data/README.md` for full disclosure of "
   "assumptions and limitations).\n"
   "2. Zone-hour demand prediction has an inherent ceiling (sparse counts); a "
   "Poisson/negative-binomial regression or aggregating to coarser time buckets could "
   "likely improve this further — noted as future work.\n"
   "3. Driver availability has a stochastic component by design; further gains would "
   "need richer per-driver behavioural features not modeled here.\n"
   "4. The dispatch simulation uses Euclidean campus distances rather than actual "
   "footpaths/roads.\n\n"
   "**Conclusion:** the project demonstrates a complete, honestly-evaluated ML "
   "lifecycle — from a documented synthetic dataset through four distinct models "
   "(regression + classification), ensembling, hyperparameter tuning, SHAP "
   "explainability, and a dispatch policy that measurably improves on the rule-based "
   "baseline — deployed as an interactive Streamlit application (`app/streamlit_app.py`) "
   "with Student, Driver, and Admin views.")

nb['cells'] = cells

out_path = Path(__file__).parent / "01_EDA_and_Modeling.ipynb"
nbf.write(nb, out_path)
print(f"Notebook structure written to {out_path} ({len(cells)} cells). Executing...")

client = NotebookClient(nb, timeout=600, kernel_name='python3',
                         resources={'metadata': {'path': str(Path(__file__).parent)}})
client.execute()
nbf.write(nb, out_path)
print("Executed and saved with outputs.")
