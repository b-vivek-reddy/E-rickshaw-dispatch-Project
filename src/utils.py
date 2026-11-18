"""Shared constants and helper functions used across preprocessing, training, and the app."""
import math
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
for d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Chronological split boundaries (see data/README.md for rationale)
TRAIN_END_DAY = 84     # days 0-83  -> train
VAL_END_DAY = 102       # days 84-101 -> validation
# days 102-119 -> test

NEIGHBOR_RADIUS_KM = 1.5   # radius used for the "nearby available drivers" feature


def load_locations():
    return pd.read_csv(RAW_DIR / "locations.csv")


def zone_distance_matrix(loc_df):
    """Return (zone_names, distance_matrix) — symmetric km distances between all zones."""
    names = loc_df["zone_name"].tolist()
    coords = loc_df[["x_km", "y_km"]].to_numpy()
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    return names, dist


def cyclical_encode(df, col, period, prefix=None):
    prefix = prefix or col
    df[f"{prefix}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{prefix}_cos"] = np.cos(2 * np.pi * df[col] / period)
    return df


def day_index(date_series, start_date):
    """Return integer day offset from the simulation start date, used for chronological splits."""
    d = pd.to_datetime(date_series)
    start = pd.to_datetime(start_date)
    return (d - start).dt.days


def chronological_split(df, day_col="day_idx"):
    train = df[df[day_col] < TRAIN_END_DAY].copy()
    val = df[(df[day_col] >= TRAIN_END_DAY) & (df[day_col] < VAL_END_DAY)].copy()
    test = df[df[day_col] >= VAL_END_DAY].copy()
    return train, val, test
