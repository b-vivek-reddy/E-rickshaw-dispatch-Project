"""
Feature Engineering
====================
Builds three analysis-ready, leakage-safe datasets from the raw simulation logs:

  1. demand_dataset        -> for Demand Prediction (regression: requests per zone-hour)
  2. availability_dataset  -> for Driver Availability Prediction (classification)
  3. eta_failed_dataset    -> for ETA Prediction (regression) and Failed-Request
                               Prediction (classification), which share the same
                               feature set since both describe "how hard is it to
                               serve this request"

Leakage safeguards:
  - All lag/rolling features are shifted so a prediction at time T only ever uses
    information available strictly before T.
  - Train/val/test splits are chronological (see utils.chronological_split), never
    random row shuffles, since this is time-ordered operational data.
"""
import numpy as np
import pandas as pd
from utils import (RAW_DIR, PROCESSED_DIR, load_locations, zone_distance_matrix,
                    cyclical_encode, day_index, NEIGHBOR_RADIUS_KM)

SIM_START = "2026-01-05"


def _load_raw():
    rides = pd.read_csv(RAW_DIR / "ride_requests.csv", parse_dates=["timestamp", "date"])
    shifts = pd.read_csv(RAW_DIR / "driver_shift_log.csv", parse_dates=["date"])
    locs = load_locations()
    return rides, shifts, locs


# ---------------------------------------------------------------------------
# Feature: nearby available-driver counts at each (date, hour_slot, zone)
# ---------------------------------------------------------------------------
def _nearby_available_drivers_table(shifts, locs):
    names, dist = zone_distance_matrix(locs)
    adjacency = (dist <= NEIGHBOR_RADIUS_KM).astype(float)  # zone x zone, 1 if within radius

    avail = shifts[shifts["status"] == "available"]
    counts = (avail.groupby(["date", "hour_slot", "home_zone"])
                    .size()
                    .unstack(fill_value=0))
    counts = counts.reindex(columns=names, fill_value=0)

    neighbor_counts = counts.to_numpy() @ adjacency.T
    neighbor_df = pd.DataFrame(neighbor_counts, index=counts.index, columns=names)
    neighbor_df = neighbor_df.reset_index()  # date, hour_slot, <zone columns...>
    return neighbor_df  # wide format: one column per zone = "available drivers near this zone"


def _nearest_quarter(hour, minute):
    total_min = hour * 60 + minute
    frame_min = total_min - 6 * 60
    if frame_min < 0:
        frame_min += 24 * 60
    q_frame = round(frame_min / 15) * 15
    return round(6 + q_frame / 60.0, 2)


# ---------------------------------------------------------------------------
# 1) DEMAND DATASET
# ---------------------------------------------------------------------------
def build_demand_dataset(rides, locs):
    df = rides.copy()
    df["day_idx"] = day_index(df["date"], SIM_START)

    agg = (df.groupby(["day_idx", "date", "hour", "pickup_zone", "weekday", "is_weekend",
                        "is_exam_period", "is_raining", "pickup_zone_type"])
             .size().reset_index(name="n_requests"))

    # Ensure every (day, hour, zone) combination exists (0 requests is a real observation)
    all_days = agg[["day_idx", "date", "weekday", "is_weekend", "is_exam_period", "is_raining"]].drop_duplicates()
    all_hours = pd.DataFrame({"hour": range(24)})
    zone_type_map = locs.set_index("zone_name")["zone_type"].to_dict()
    all_zones = pd.DataFrame({"pickup_zone": locs["zone_name"]})
    all_zones["pickup_zone_type"] = all_zones["pickup_zone"].map(zone_type_map)

    skeleton = all_days.merge(all_hours, how="cross").merge(all_zones, how="cross")
    full = skeleton.merge(agg, on=["day_idx", "date", "hour", "pickup_zone", "weekday",
                                    "is_weekend", "is_exam_period", "is_raining",
                                    "pickup_zone_type"], how="left")
    full["n_requests"] = full["n_requests"].fillna(0).astype(int)

    full = full.sort_values(["pickup_zone", "day_idx", "hour"]).reset_index(drop=True)

    # Lag / rolling features per zone (shifted -> no leakage)
    full["lag_1h_same_zone"] = full.groupby("pickup_zone")["n_requests"].shift(1)
    full["lag_24h_same_zone_hour"] = full.groupby(["pickup_zone", "hour"])["n_requests"].shift(1)
    full["rolling7_avg_zone_hour"] = (
        full.groupby(["pickup_zone", "hour"])["n_requests"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    )
    full[["lag_1h_same_zone", "lag_24h_same_zone_hour", "rolling7_avg_zone_hour"]] = (
        full[["lag_1h_same_zone", "lag_24h_same_zone_hour", "rolling7_avg_zone_hour"]].fillna(0)
    )

    full = cyclical_encode(full, "hour", 24)
    full = cyclical_encode(full, "weekday", 7)

    full.to_csv(PROCESSED_DIR / "demand_dataset.csv", index=False)
    return full


# ---------------------------------------------------------------------------
# 2) DRIVER AVAILABILITY DATASET
# ---------------------------------------------------------------------------
def build_availability_dataset(shifts):
    df = shifts.copy()
    df["day_idx"] = day_index(df["date"], SIM_START)
    df = df[df["status"] != "off_shift"].copy()   # only predict among on-shift drivers
    df["is_available"] = (df["status"] == "available").astype(int)

    df = df.sort_values(["driver_id", "day_idx", "hour_slot"]).reset_index(drop=True)
    df["prev_slot_available"] = df.groupby("driver_id")["is_available"].shift(1)
    df["prev_slot_available"] = df["prev_slot_available"].fillna(df["is_available"].mean())

    df["hour_int"] = df["hour_slot"].astype(int) % 24
    df = cyclical_encode(df, "hour_int", 24, prefix="hour")

    df.to_csv(PROCESSED_DIR / "availability_dataset.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 3) ETA + FAILED-REQUEST DATASET
# ---------------------------------------------------------------------------
def build_eta_failed_dataset(rides, shifts, locs):
    df = rides.copy()
    df["day_idx"] = day_index(df["date"], SIM_START)
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    neighbor_df = _nearby_available_drivers_table(shifts, locs)
    neighbor_df["date_str"] = neighbor_df["date"].dt.strftime("%Y-%m-%d")

    df["slot"] = df.apply(lambda r: _nearest_quarter(r["hour"], r["minute"]), axis=1)

    neighbor_long = neighbor_df.melt(
        id_vars=["date_str", "hour_slot"], value_vars=locs["zone_name"].tolist(),
        var_name="zone", value_name="nearby_available_drivers"
    )

    df = df.merge(
        neighbor_long, left_on=["date_str", "slot", "pickup_zone"],
        right_on=["date_str", "hour_slot", "zone"], how="left"
    )
    df["nearby_available_drivers"] = df["nearby_available_drivers"].fillna(0)
    df = df.drop(columns=["hour_slot", "zone"])

    # Historical zone-level average wait time (rolling, shifted -> no leakage)
    df = df.sort_values(["pickup_zone", "timestamp"]).reset_index(drop=True)
    df["rolling_zone_wait_avg"] = (
        df.groupby("pickup_zone")["wait_time_minutes"]
          .transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    )
    overall_mean_wait = df["wait_time_minutes"].mean()
    df["rolling_zone_wait_avg"] = df["rolling_zone_wait_avg"].fillna(overall_mean_wait)

    df["rolling_zone_fail_rate"] = (
        df.groupby("pickup_zone")["request_failed"]
          .transform(lambda s: s.shift(1).expanding(min_periods=5).mean())
    )
    overall_fail_rate = df["request_failed"].mean()
    df["rolling_zone_fail_rate"] = df["rolling_zone_fail_rate"].fillna(overall_fail_rate)

    df = cyclical_encode(df, "hour", 24)
    df = cyclical_encode(df, "weekday", 7)

    df.to_csv(PROCESSED_DIR / "eta_failed_dataset.csv", index=False)
    return df


def main():
    rides, shifts, locs = _load_raw()
    print("Building demand dataset...")
    demand = build_demand_dataset(rides, locs)
    print(f"  -> {demand.shape}")

    print("Building driver availability dataset...")
    avail = build_availability_dataset(shifts)
    print(f"  -> {avail.shape}")

    print("Building ETA / failed-request dataset...")
    eta = build_eta_failed_dataset(rides, shifts, locs)
    print(f"  -> {eta.shape}")

    print("\nSaved all processed datasets to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
