"""
Model 2: DRIVER AVAILABILITY PREDICTION (classification)
Target: is_available -> will this on-shift driver be 'available' (vs 'busy') at this
        15-minute slot?

Pipeline: baseline (majority class / Logistic Regression) -> Random Forest -> XGBoost
          -> tuned XGBoost -> Stacking ensemble, with SHAP explainability.
"""
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

from utils import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, chronological_split
from model_utils import (classification_metrics, plot_classification_comparison,
                          plot_confusion, plot_roc, save_json, plot_feature_importance)

NUMERIC_FEATURES = ["hour_sin", "hour_cos", "prev_slot_available"]
BOOL_FEATURES = ["is_weekend", "is_exam_period", "is_raining"]
CAT_FEATURES = ["shift_type", "home_zone"]
TARGET = "is_available"


def load_splits():
    df = pd.read_csv(PROCESSED_DIR / "availability_dataset.csv")
    for c in BOOL_FEATURES:
        df[c] = df[c].astype(int)
    train, val, test = chronological_split(df)
    print(f"Train={len(train)}  Val={len(val)}  Test={len(test)}")
    return train, val, test


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES + BOOL_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])


def Xy(df):
    return df[NUMERIC_FEATURES + BOOL_FEATURES + CAT_FEATURES], df[TARGET]


def main():
    train, val, test = load_splits()
    X_train, y_train = Xy(train)
    X_val, y_val = Xy(val)
    X_test, y_test = Xy(test)

    pre = make_preprocessor()

    candidates = {
        "Baseline (Majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1, eval_metric="logloss"),
    }

    val_results = {}
    fitted_pipes = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        proba = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else None
        val_results[name] = classification_metrics(y_val, preds, proba)
        fitted_pipes[name] = pipe
        print(name, val_results[name])

    # ---- Hyperparameter tuning (RandomizedSearchCV, time-aware CV) on XGBoost ----
    print("\nTuning XGBoost hyperparameters...")
    param_dist = {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.02, 0.05, 0.08, 0.12],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
    }
    tune_pipe = Pipeline([("pre", pre), ("model", XGBClassifier(random_state=42, n_jobs=-1,
                                                                  eval_metric="logloss"))])
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(tune_pipe, param_dist, n_iter=15, cv=tscv,
                                 scoring="f1", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    preds = best_xgb.predict(X_val)
    proba = best_xgb.predict_proba(X_val)[:, 1]
    val_results["XGBoost (Tuned)"] = classification_metrics(y_val, preds, proba)
    fitted_pipes["XGBoost (Tuned)"] = best_xgb
    print("Best params:", search.best_params_)
    print("XGBoost (Tuned)", val_results["XGBoost (Tuned)"])

    # ---- Ensemble: Stacking (RF + tuned XGBoost -> Logistic meta-learner) ----
    print("\nTraining Stacking ensemble (RF + tuned XGBoost -> Logistic Regression)...")
    best_xgb_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    stack = StackingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
            ("xgb", XGBClassifier(**best_xgb_params, random_state=42, n_jobs=-1, eval_metric="logloss")),
        ],
        final_estimator=LogisticRegression(max_iter=500), cv=3, n_jobs=-1,
    )
    stack_pipe = Pipeline([("pre", pre), ("model", stack)])
    stack_pipe.fit(X_train, y_train)
    preds = stack_pipe.predict(X_val)
    proba = stack_pipe.predict_proba(X_val)[:, 1]
    val_results["Stacking Ensemble"] = classification_metrics(y_val, preds, proba)
    fitted_pipes["Stacking Ensemble"] = stack_pipe
    print("Stacking Ensemble", val_results["Stacking Ensemble"])

    plot_classification_comparison(val_results, "Driver Availability — Model Comparison (Validation)",
                                    FIGURES_DIR / "availability_model_comparison.png")

    # ---- Pick best model by validation F1, refit on train+val, evaluate once on test ----
    best_name = max(val_results, key=lambda n: val_results[n]["F1"])
    print(f"\nBest model on validation: {best_name}")

    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    final_model = fitted_pipes[best_name]
    final_model.fit(X_trval, y_trval)
    test_preds = final_model.predict(X_test)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = classification_metrics(y_test, test_preds, test_proba)
    print("TEST metrics (held-out, final):", test_metrics)

    plot_confusion(y_test, test_preds, f"Driver Availability — {best_name} Confusion Matrix (Test)",
                   FIGURES_DIR / "availability_confusion.png", labels=("Busy", "Available"))
    plot_roc(y_test, test_proba, f"Driver Availability — {best_name} ROC Curve (Test)",
             FIGURES_DIR / "availability_roc.png")

    # ---- Feature importance / SHAP ----
    importance_model = fitted_pipes["XGBoost (Tuned)"]
    feat_names = (NUMERIC_FEATURES + BOOL_FEATURES +
                  list(importance_model.named_steps["pre"].named_transformers_["cat"]
                       .get_feature_names_out(CAT_FEATURES)))
    importances = importance_model.named_steps["model"].feature_importances_
    plot_feature_importance(importances, feat_names,
                             "Driver Availability — Feature Importance (XGBoost)",
                             FIGURES_DIR / "availability_feature_importance.png")

    print("Computing SHAP values (this may take a moment)...")
    X_test_transformed = importance_model.named_steps["pre"].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    sample = X_test_transformed[:800]
    explainer = shap.TreeExplainer(importance_model.named_steps["model"])
    shap_values = explainer.shap_values(sample)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, sample, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "availability_shap_summary.png", dpi=130)
    plt.close()

    # ---- Save artifacts ----
    joblib.dump(final_model, MODELS_DIR / "availability_model.joblib")
    save_json({"validation_results": val_results, "test_metrics": test_metrics,
               "best_model": best_name, "best_xgb_params": search.best_params_},
              MODELS_DIR / "availability_model_metrics.json")
    print("\nSaved availability_model.joblib and availability_model_metrics.json")


if __name__ == "__main__":
    main()
