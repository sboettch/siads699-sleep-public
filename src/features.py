"""
features.py
-----------
Feature engineering for sleep health analysis.
Computes participant-level and night-level features from cleaned sleep
and activity data. Designed to work identically with synthetic data
(local dev) and real All of Us data (Workbench).
"""

import pandas as pd
import numpy as np


# ── Sleep cleaning ─────────────────────────────────────────────────────────────

def clean_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inclusion/exclusion criteria from Patten et al. (2026):
      - Keep only main sleep records (is_main_sleep == True)
      - Exclude nights with < 4 hours of sleep
      - Exclude participants where >= 30% of nights are < 4h
      - Require >= 4 valid nights per participant

    Parameters
    ----------
    df : pd.DataFrame
        Raw sleep data with columns:
        person_id, sleep_date, is_main_sleep, minute_asleep

    Returns
    -------
    pd.DataFrame
        Cleaned sleep data
    """
    df = df.copy()
    df["sleep_date"] = pd.to_datetime(df["sleep_date"])
    df["hours_asleep"] = df["minute_asleep"] / 60

    # 1. Keep only main sleep (drop naps)
    df = df[df["is_main_sleep"] == True].copy()

    # 2. Flag valid nights (>= 4 hours)
    df["valid_night"] = df["hours_asleep"] >= 4

    # 3. Exclude participants with >= 30% invalid nights
    validity_rate = df.groupby("person_id")["valid_night"].mean()
    valid_pids = validity_rate[validity_rate >= 0.70].index
    df = df[df["person_id"].isin(valid_pids)]

    # 4. Keep only valid nights
    df = df[df["valid_night"]].copy()

    # 5. Require >= 4 valid nights per participant
    night_counts = df.groupby("person_id").size()
    pids_enough = night_counts[night_counts >= 4].index
    df = df[df["person_id"].isin(pids_enough)]

    return df.drop(columns=["valid_night"])


def clean_steps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply inclusion/exclusion criteria for activity data:
      - Steps > 100 (remove non-wear)
      - Steps < 100,000 (remove impossible values)
      - Require >= 4 valid days per participant

    Parameters
    ----------
    df : pd.DataFrame
        Raw steps data with columns: person_id, date, steps

    Returns
    -------
    pd.DataFrame
        Cleaned steps data
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Valid day: 100 <= steps < 100,000
    df = df[(df["steps"] >= 100) & (df["steps"] < 100_000)]

    # Require >= 4 valid days
    day_counts = df.groupby("person_id").size()
    pids_enough = day_counts[day_counts >= 4].index
    df = df[df["person_id"].isin(pids_enough)]

    return df


# ── Participant-level feature engineering ──────────────────────────────────────

def make_participant_features(
    sleep_df: pd.DataFrame,
    steps_df: pd.DataFrame,
    demo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate night/day level data to participant-level features.
    These are used as inputs to the predictive model.

    NOTE: All aggregation happens here — no row-level data is used
    downstream of this function, which keeps cross-validation clean
    and ensures no night-level leakage.

    Returns
    -------
    pd.DataFrame
        One row per participant with engineered features
    """
    # ── Sleep features ────────────────────────────────────────────────────────
    sleep_feats = sleep_df.groupby("person_id").agg(
        mean_sleep_hrs    = ("hours_asleep", "mean"),
        std_sleep_hrs     = ("hours_asleep", "std"),       # consistency
        median_sleep_hrs  = ("hours_asleep", "median"),
        min_sleep_hrs     = ("hours_asleep", "min"),
        n_valid_nights    = ("hours_asleep", "count"),
        pct_short_sleep   = ("hours_asleep", lambda x: (x < 6).mean()),  # < 6h
        pct_long_sleep    = ("hours_asleep", lambda x: (x > 9).mean()),  # > 9h
    ).reset_index()

    # Sleep regularity: IQR of sleep duration (lower = more consistent)
    sleep_iqr = sleep_df.groupby("person_id")["hours_asleep"].quantile([0.25, 0.75]).unstack()
    sleep_iqr.columns = ["q25_sleep", "q75_sleep"]
    sleep_iqr["iqr_sleep_hrs"] = sleep_iqr["q75_sleep"] - sleep_iqr["q25_sleep"]
    sleep_iqr = sleep_iqr[["iqr_sleep_hrs"]].reset_index()

    sleep_feats = sleep_feats.merge(sleep_iqr, on="person_id", how="left")

    # ── Activity features ─────────────────────────────────────────────────────
    steps_feats = steps_df.groupby("person_id").agg(
        mean_daily_steps  = ("steps", "mean"),
        median_daily_steps= ("steps", "median"),
        std_daily_steps   = ("steps", "std"),
        n_valid_days      = ("steps", "count"),
    ).reset_index()

    # ── Join everything ───────────────────────────────────────────────────────
    features = (
        demo_df
        .merge(sleep_feats, on="person_id", how="inner")
        .merge(steps_feats, on="person_id", how="left")
    )

    return features


# ── Target variables ───────────────────────────────────────────────────────────

def add_targets(features: pd.DataFrame) -> pd.DataFrame:
    """
    Define prediction targets.
      - target_sleep_duration : mean nightly sleep hours
      - target_sleep_consistency : std of nightly sleep hours (lower = better)
    """
    df = features.copy()
    df["target_sleep_duration"]    = df["mean_sleep_hrs"]
    df["target_sleep_consistency"] = df["std_sleep_hrs"]
    return df
