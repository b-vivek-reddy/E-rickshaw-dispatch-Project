"""Shared helpers for model evaluation, comparison, and explainability, reused by all
three training scripts (demand, availability, eta_failed)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve)

sns.set_style("whitegrid")


# ------------------------------- Regression ---------------------------------
def regression_metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def plot_regression_comparison(results: dict, title, path):
    """results: {model_name: {'MAE':.., 'RMSE':.., 'R2':..}}"""
    names = list(results.keys())
    mae = [results[n]["MAE"] for n in names]
    rmse = [results[n]["RMSE"] for n in names]
    r2 = [results[n]["R2"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, label, color in zip(axes, [mae, rmse, r2], ["MAE", "RMSE", "R2"],
                                       ["#4C72B0", "#DD8452", "#55A868"]):
        ax.bar(names, vals, color=color)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residuals(y_true, y_pred, title, path):
    resid = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_pred, resid, alpha=0.25, s=10)
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Predicted")
    sns.histplot(resid, bins=40, ax=axes[1], kde=True, color="#4C72B0")
    axes[1].set_title("Residual Distribution")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------ Classification -------------------------------
def classification_metrics(y_true, y_pred, y_proba=None):
    m = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            m["ROC_AUC"] = round(roc_auc_score(y_true, y_proba), 4)
        except ValueError:
            m["ROC_AUC"] = None
    return m


def plot_classification_comparison(results: dict, title, path):
    names = list(results.keys())
    metrics_order = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.15
    x = np.arange(len(names))
    for i, m in enumerate(metrics_order):
        vals = [results[n].get(m) or 0 for n in names]
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=20)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, path, labels=("Negative", "Positive")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(y_true, y_proba, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------- Generic -------------------------------------
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def plot_feature_importance(importances, feature_names, title, path, top_n=15):
    order = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1], [importances[i] for i in order][::-1],
            color="#4C72B0")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
