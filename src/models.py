"""
models.py
---------
Modeling pipeline for sleep health analysis.

Includes:
  - Baseline model (predict population mean)
  - Regularized regression (Ridge/Lasso)
  - Random Forest
  - Participant-level cross-validation (no night-level leakage)
  - Fairness evaluation across demographic subgroups
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# ── Feature / target setup ────────────────────────────────────────────────────

FEATURE_COLS = [
    "age", "bmi",
    "mean_daily_steps", "median_daily_steps", "std_daily_steps",
    "n_valid_nights", "n_valid_days",
    "pct_short_sleep", "pct_long_sleep",
    "iqr_sleep_hrs",
]

CATEGORICAL_COLS = ["gender", "race", "income_tier", "region"]


def prepare_X_y(features_df: pd.DataFrame, target: str) -> tuple:
    """
    Prepare feature matrix and target vector.

    Parameters
    ----------
    features_df : pd.DataFrame
        Participant-level features (one row per participant)
    target : str
        Target column name, e.g. 'target_sleep_duration'

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = features_df.dropna(subset=[target]).copy()

    # One-hot encode categoricals
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    feature_cols = [c for c in df.columns
                    if c in FEATURE_COLS
                    or any(c.startswith(cat + "_") for cat in CATEGORICAL_COLS)]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target]

    return X, y


# ── Models ────────────────────────────────────────────────────────────────────

def get_models() -> dict:
    """Return dict of named model pipelines."""
    return {
        "Baseline (mean)": None,   # handled separately
        "Ridge":           Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Lasso":           Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01))]),
        "Random Forest":   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boost":  GradientBoostingRegressor(n_estimators=100, random_state=42),
    }


# ── Participant-level cross-validation ────────────────────────────────────────

def participant_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_splits: int = 5,
) -> dict:
    """
    Cross-validate at the PARTICIPANT level.
    Each participant appears in either train OR test — never both.
    This prevents leakage from within-person correlation.

    Returns dict with mean RMSE, MAE, R² across folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores, r2_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_scores.append(mean_absolute_error(y_test, preds))
        r2_scores.append(model.score(X_test, y_test))

    return {
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std":  np.std(rmse_scores),
        "mae_mean":  np.mean(mae_scores),
        "r2_mean":   np.mean(r2_scores),
    }


def baseline_cv(y: pd.Series, n_splits: int = 5) -> dict:
    """Naive baseline: always predict training set mean."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, mae_scores = [], []

    for train_idx, test_idx in kf.split(y):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        preds = np.full(len(y_test), y_train.mean())
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
        mae_scores.append(mean_absolute_error(y_test, preds))

    return {
        "rmse_mean": np.mean(rmse_scores),
        "rmse_std":  np.std(rmse_scores),
        "mae_mean":  np.mean(mae_scores),
        "r2_mean":   0.0,
    }


def run_all_models(features_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Run all models with participant-level CV and return results table.
    """
    X, y = prepare_X_y(features_df, target)
    models = get_models()
    results = []

    for name, model in models.items():
        print(f"  Running {name}...")
        if model is None:
            scores = baseline_cv(y)
        else:
            scores = participant_cv(X, y, model)
        results.append({"Model": name, **scores})

    return pd.DataFrame(results).sort_values("rmse_mean")


# ── Fairness evaluation ────────────────────────────────────────────────────────

def fairness_eval(
    features_df: pd.DataFrame,
    target: str,
    model,
    subgroup_col: str,
) -> pd.DataFrame:
    """
    Evaluate model performance separately for each subgroup.
    Flags groups with RMSE > 1.25x the overall RMSE.

    Parameters
    ----------
    subgroup_col : str
        Column to split on, e.g. 'race', 'gender', 'income_tier'
    """
    X, y = prepare_X_y(features_df, target)
    model.fit(X, y)
    preds = model.predict(X)

    results = []
    overall_rmse = np.sqrt(mean_squared_error(y, preds))

    for group in features_df[subgroup_col].dropna().unique():
        mask = features_df[subgroup_col] == group
        if mask.sum() < 10:
            continue
        g_rmse = np.sqrt(mean_squared_error(y[mask], preds[mask]))
        results.append({
            "subgroup":     group,
            "n":            mask.sum(),
            "rmse":         round(g_rmse, 4),
            "rmse_ratio":   round(g_rmse / overall_rmse, 3),
            "flagged":      g_rmse > 1.25 * overall_rmse,
        })

    return pd.DataFrame(results).sort_values("rmse_ratio", ascending=False)
