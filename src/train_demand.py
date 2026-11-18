"""
Model 1: DEMAND PREDICTION (regression)
Target: n_requests  -> number of ride requests in a given (zone, hour) in the next hour.

Pipeline: baseline (mean/Linear Regression) -> RandomForest -> XGBoost -> tuned XGBoost
          -> Voting ensemble (RF + XGBoost), with SHAP explainability on the best model.
"""
import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

from utils import PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, chronological_split
from model_utils import (regression_metrics, plot_regression_comparison, plot_residuals,
                          save_json, plot_feature_importance)

NUMERIC_FEATURES = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                     "lag_1h_same_zone", "lag_24h_same_zone_hour", "rolling7_avg_zone_hour"]
BOOL_FEATURES = ["is_weekend", "is_exam_period", "is_raining"]
CAT_FEATURES = ["pickup_zone", "pickup_zone_type"]
TARGET = "n_requests"


def load_splits():
    df = pd.read_csv(PROCESSED_DIR / "demand_dataset.csv")
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
    # Train models on train+val features together only after model selection on val;
    # for final test evaluation we refit on train+val and evaluate once on test.
    X_train, y_train = Xy(train)
    X_val, y_val = Xy(val)
    X_test, y_test = Xy(test)

    pre = make_preprocessor()

    candidates = {
        "Baseline (Mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 n_jobs=-1, objective="reg:squarederror"),
    }

    val_results = {}
    fitted_pipes = {}
    for name, model in candidates.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = np.clip(pipe.predict(X_val), 0, None)
        val_results[name] = regression_metrics(y_val, preds)
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
    tune_pipe = Pipeline([("pre", pre), ("model", XGBRegressor(random_state=42, n_jobs=-1,
                                                                 objective="reg:squarederror"))])
    tscv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(tune_pipe, param_dist, n_iter=15, cv=tscv,
                                 scoring="neg_mean_absolute_error", random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    preds = np.clip(best_xgb.predict(X_val), 0, None)
    val_results["XGBoost (Tuned)"] = regression_metrics(y_val, preds)
    fitted_pipes["XGBoost (Tuned)"] = best_xgb
    print("Best params:", search.best_params_)
    print("XGBoost (Tuned)", val_results["XGBoost (Tuned)"])

    # ---- Ensemble: Voting (RF + tuned XGBoost) ----
    print("\nTraining Voting ensemble (RF + tuned XGBoost)...")
    voting = VotingRegressor([
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
        ("xgb", XGBRegressor(**{k.replace("model__", ""): v for k, v in search.best_params_.items()},
                              random_state=42, n_jobs=-1, objective="reg:squarederror")),
    ])
    ens_pipe = Pipeline([("pre", pre), ("model", voting)])
    ens_pipe.fit(X_train, y_train)
    preds = np.clip(ens_pipe.predict(X_val), 0, None)
    val_results["Voting Ensemble"] = regression_metrics(y_val, preds)
    fitted_pipes["Voting Ensemble"] = ens_pipe
    print("Voting Ensemble", val_results["Voting Ensemble"])

    plot_regression_comparison(val_results, "Demand Prediction — Model Comparison (Validation)",
                                FIGURES_DIR / "demand_model_comparison.png")

    # ---- Pick best model by validation MAE, refit on train+val, evaluate once on test ----
    best_name = min(val_results, key=lambda n: val_results[n]["MAE"])
    print(f"\nBest model on validation: {best_name}")

    X_trval = pd.concat([X_train, X_val]); y_trval = pd.concat([y_train, y_val])
    final_model = fitted_pipes[best_name]
    final_model.fit(X_trval, y_trval)
    test_preds = np.clip(final_model.predict(X_test), 0, None)
    test_metrics = regression_metrics(y_test, test_preds)
    print("TEST metrics (held-out, final):", test_metrics)

    plot_residuals(y_test, test_preds, f"Demand Prediction — {best_name} Residuals (Test)",
                    FIGURES_DIR / "demand_residuals.png")

    # ---- Feature importance / SHAP (on the tree model regardless of which ensemble won) ----
    importance_model = fitted_pipes["XGBoost (Tuned)"]
    feat_names = (NUMERIC_FEATURES + BOOL_FEATURES +
                  list(importance_model.named_steps["pre"].named_transformers_["cat"]
                       .get_feature_names_out(CAT_FEATURES)))
    importances = importance_model.named_steps["model"].feature_importances_
    plot_feature_importance(importances, feat_names, "Demand Prediction — Feature Importance (XGBoost)",
                             FIGURES_DIR / "demand_feature_importance.png")

    print("Computing SHAP values (this may take a moment)...")
    X_test_transformed = importance_model.named_steps["pre"].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()
    sample = X_test_transformed[:800]  # subsample for speed
    explainer = shap.TreeExplainer(importance_model.named_steps["model"])
    shap_values = explainer.shap_values(sample)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, sample, feature_names=feat_names, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "demand_shap_summary.png", dpi=130)
    plt.close()

    # ---- Save artifacts ----
    joblib.dump(final_model, MODELS_DIR / "demand_model.joblib")
    save_json({"validation_results": val_results, "test_metrics": test_metrics,
               "best_model": best_name, "best_xgb_params": search.best_params_},
              MODELS_DIR / "demand_model_metrics.json")
    print("\nSaved demand_model.joblib and demand_model_metrics.json")


if __name__ == "__main__":
    main()
