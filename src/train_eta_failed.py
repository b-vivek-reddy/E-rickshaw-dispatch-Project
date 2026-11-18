"""
Model 3: ETA / WAIT-TIME PREDICTION (regression) — trained on successfully matched requests.
Model 4: FAILED-REQUEST PREDICTION (classification) — trained on ALL requests, predicts
          whether a request will fail to find a driver at all. This directly feeds the
          Dynamic Search Radius decision in src/dispatch.py: if predicted failure risk is
          high, the dispatcher starts with a wider radius immediately instead of waiting
          to exhaust the default radius first.

Both share the same feature set (how "hard" is this request to serve), so they are
trained together in this script using the same ColumnTransformer.
"""
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               VotingRegressor, StackingClassifier)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier

from utils import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, chronological_split
from model_utils import (regression_metrics, classification_metrics,
                          plot_regression_comparison, plot_residuals,
                          plot_classification_comparison, plot_confusion, plot_roc,
                          save_json, plot_feature_importance)

NUMERIC_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "distance_km",
                     "nearby_available_drivers", "rolling_zone_wait_avg", "rolling_zone_fail_rate"]
BOOL_FEATURES = ["is_weekend", "is_exam_period", "is_raining"]
CAT_FEATURES = ["pickup_zone_type", "dropoff_zone_type", "time_block"]


def load_data():
    df = pd.read_csv(PROCESSED_DIR / "eta_failed_dataset.csv")
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(int)
    df["request_failed"] = df["request_failed"].astype(int)
    train, val, test = chronological_split(df)
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES + BOOL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def Xy(df, target):
    feats = NUMERIC_FEATURES + BOOL_FEATURES + CAT_FEATURES
    return df[feats], df[target]


# =============================================================================
# PART A: ETA / WAIT-TIME REGRESSION (successful requests only)
# =============================================================================
def train_eta(train, val, test):
    print("\n" + "=" * 70)
    print("MODEL 3: ETA / WAIT-TIME PREDICTION")
    print("=" * 70)
    tr = train[train["request_failed"] == 0]
    va = val[val["request_failed"] == 0]
    te = test[test["request_failed"] == 0]

    X_train, y_train = Xy(tr, "wait_time_minutes")
    X_val, y_val = Xy(va, "wait_time_minutes")
    X_test, y_test = Xy(te, "wait_time_minutes")

    pre = make_preprocessor()
    candidates = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
    }
    val_results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_val), 0, None)
        val_results[name] = regression_metrics(y_val, preds)
        fitted[name] = pipe
        print(name, val_results[name])

    print("\nTuning XGBoost...")
    param_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }
    tune_pipe = Pipeline([("pre", pre), ("model", XGBRegressor(random_state=42, n_jobs=-1))])
    search = RandomizedSearchCV(tune_pipe, param_dist, n_iter=15, cv=TimeSeriesSplit(n_splits=4),
                                 scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    fitted["XGBoost (Tuned)"] = search.best_estimator_
    preds = np.clip(search.best_estimator_.predict(X_val), 0, None)
    val_results["XGBoost (Tuned)"] = regression_metrics(y_val, preds)
    print("Best params:", search.best_params_)
    print("XGBoost (Tuned)", val_results["XGBoost (Tuned)"])

    print("\nVoting ensemble (RF + tuned XGBoost)...")
    best_xgb_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    voting = VotingRegressor([
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(**best_xgb_params, random_state=42, n_jobs=-1)),
    ])
    ens_pipe = Pipeline([("pre", pre), ("model", voting)]).fit(X_train, y_train)
    preds = np.clip(ens_pipe.predict(X_val), 0, None)
    val_results["Voting Ensemble"] = regression_metrics(y_val, preds)
    fitted["Voting Ensemble"] = ens_pipe
    print("Voting Ensemble", val_results["Voting Ensemble"])

    plot_regression_comparison(val_results, "ETA Prediction — Model Comparison (Validation)",
                                FIGURES_DIR / "eta_model_comparison.png")

    best_name = min(val_results, key=lambda n: val_results[n]["MAE"])
    print(f"Best model: {best_name}")
    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    final_model = fitted[best_name].fit(X_trval, y_trval)
    test_preds = np.clip(final_model.predict(X_test), 0, None)
    test_metrics = regression_metrics(y_test, test_preds)
    print("TEST metrics:", test_metrics)

    plot_residuals(y_test, test_preds, f"ETA Prediction — {best_name} Residuals (Test)",
                    FIGURES_DIR / "eta_residuals.png")

    imp_model = fitted["XGBoost (Tuned)"]
    feat_names = (NUMERIC_FEATURES + BOOL_FEATURES +
                  list(imp_model.named_steps["pre"].named_transformers_["cat"]
                       .get_feature_names_out(CAT_FEATURES)))
    plot_feature_importance(imp_model.named_steps["model"].feature_importances_, feat_names,
                             "ETA Prediction — Feature Importance (XGBoost)",
                             FIGURES_DIR / "eta_feature_importance.png")

    print("Computing SHAP values...")
    Xt = imp_model.named_steps["pre"].transform(X_test)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    sample = Xt[:800]
    explainer = shap.TreeExplainer(imp_model.named_steps["model"])
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "eta_shap_summary.png", dpi=130); plt.close()

    joblib.dump(final_model, MODELS_DIR / "eta_model.joblib")
    save_json({"validation_results": val_results, "test_metrics": test_metrics,
               "best_model": best_name, "best_xgb_params": search.best_params_},
              MODELS_DIR / "eta_model_metrics.json")
    print("Saved eta_model.joblib")


# =============================================================================
# PART B: FAILED-REQUEST CLASSIFICATION (all requests)
# =============================================================================
def train_failed(train, val, test):
    print("\n" + "=" * 70)
    print("MODEL 4: FAILED-REQUEST PREDICTION")
    print("=" * 70)
    X_train, y_train = Xy(train, "request_failed")
    X_val, y_val = Xy(val, "request_failed")
    X_test, y_test = Xy(test, "request_failed")

    pre = make_preprocessor()
    candidates = {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42,
                                                  n_jobs=-1, class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1, eval_metric="logloss",
                                  scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1)),
    }
    val_results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)]).fit(X_train, y_train)
        preds = pipe.predict(X_val)
        proba = pipe.predict_proba(X_val)[:, 1]
        val_results[name] = classification_metrics(y_val, preds, proba)
        fitted[name] = pipe
        print(name, val_results[name])

    print("\nTuning XGBoost...")
    param_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    tune_pipe = Pipeline([("pre", pre), ("model", XGBClassifier(random_state=42, n_jobs=-1,
                                                                  eval_metric="logloss",
                                                                  scale_pos_weight=spw))])
    search = RandomizedSearchCV(tune_pipe, param_dist, n_iter=15, cv=TimeSeriesSplit(n_splits=4),
                                 scoring="f1", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    fitted["XGBoost (Tuned)"] = search.best_estimator_
    preds = search.best_estimator_.predict(X_val)
    proba = search.best_estimator_.predict_proba(X_val)[:, 1]
    val_results["XGBoost (Tuned)"] = classification_metrics(y_val, preds, proba)
    print("Best params:", search.best_params_)
    print("XGBoost (Tuned)", val_results["XGBoost (Tuned)"])

    print("\nStacking ensemble (RF + tuned XGBoost -> Logistic)...")
    best_xgb_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=250, max_depth=8, random_state=42,
                                           n_jobs=-1, class_weight="balanced")),
            ("xgb", XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1,
                                   eval_metric="logloss", scale_pos_weight=spw)),
        ],
        final_estimator=LogisticRegression(max_iter=500, class_weight="balanced"), cv=3, n_jobs=-1,
    )
    stack_pipe = Pipeline([("pre", pre), ("model", stack)]).fit(X_train, y_train)
    preds = stack_pipe.predict(X_val)
    proba = stack_pipe.predict_proba(X_val)[:, 1]
    val_results["Stacking Ensemble"] = classification_metrics(y_val, preds, proba)
    fitted["Stacking Ensemble"] = stack_pipe
    print("Stacking Ensemble", val_results["Stacking Ensemble"])

    plot_classification_comparison(val_results, "Failed-Request Prediction — Model Comparison (Validation)",
                                    FIGURES_DIR / "failed_model_comparison.png")

    best_name = max(val_results, key=lambda n: val_results[n]["F1"])
    print(f"Best model: {best_name}")
    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    final_model = fitted[best_name].fit(X_trval, y_trval)
    test_preds = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_preds, test_proba)
    print("TEST metrics:", test_metrics)

    plot_confusion(y_test, test_preds, f"Failed-Request — {best_name} Confusion Matrix (Test)",
                   FIGURES_DIR / "failed_confusion.png", labels=("Fulfilled", "Failed"))
    plot_roc(y_test, test_proba, f"Failed-Request — {best_name} ROC Curve (Test)",
             FIGURES_DIR / "failed_roc.png")

    imp_model = fitted["XGBoost (Tuned)"]
    feat_names = (NUMERIC_FEATURES + BOOL_FEATURES +
                  list(imp_model.named_steps["pre"].named_transformers_["cat"]
                       .get_feature_names_out(CAT_FEATURES)))
    plot_feature_importance(imp_model.named_steps["model"].feature_importances_, feat_names,
                             "Failed-Request Prediction — Feature Importance (XGBoost)",
                             FIGURES_DIR / "failed_feature_importance.png")

    print("Computing SHAP values...")
    Xt = imp_model.named_steps["pre"].transform(X_test)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    sample = Xt[:800]
    explainer = shap.TreeExplainer(imp_model.named_steps["model"])
    shap_values = explainer.shap_values(sample)
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(sv, sample, feature_names=feat_names, show=False)
    plt.tight_layout(); plt.savefig(FIGURES_DIR / "failed_shap_summary.png", dpi=130); plt.close()

    joblib.dump(final_model, MODELS_DIR / "failed_model.joblib")
    save_json({"validation_results": val_results, "test_metrics": test_metrics,
               "best_model": best_name, "best_xgb_params": search.best_params_},
              MODELS_DIR / "failed_model_metrics.json")
    print("Saved failed_model.joblib")


def main():
    train, val, test = load_data()
    train_eta(train, val, test)
    train_failed(train, val, test)


if __name__ == "__main__":
    main()
