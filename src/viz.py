"""
viz.py
------
Reusable plotting functions for sleep health analysis.
All functions accept DataFrames and return matplotlib/plotly figures
so they work identically locally and in the Workbench.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Set2
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── EDA plots ─────────────────────────────────────────────────────────────────

def plot_sleep_distribution(sleep_df: pd.DataFrame) -> go.Figure:
    """Histogram of nightly sleep duration across all valid nights."""
    fig = px.histogram(
        sleep_df,
        x="hours_asleep",
        nbins=60,
        labels={"hours_asleep": "Hours Asleep"},
        title="Distribution of Nightly Sleep Duration",
        color_discrete_sequence=[PALETTE[0]],
    )
    fig.add_vline(x=7, line_dash="dash", line_color="gray",
                  annotation_text="7h guideline", annotation_position="top right")
    fig.update_layout(bargap=0.05)
    return fig


def plot_sleep_by_group(features_df: pd.DataFrame, group_col: str) -> go.Figure:
    """Boxplot of mean sleep duration by demographic group."""
    fig = px.box(
        features_df,
        x=group_col,
        y="mean_sleep_hrs",
        color=group_col,
        labels={"mean_sleep_hrs": "Mean Sleep (hrs)", group_col: group_col.title()},
        title=f"Mean Sleep Duration by {group_col.title()}",
        color_discrete_sequence=PALETTE,
    )
    fig.add_hline(y=7, line_dash="dash", line_color="gray")
    fig.update_layout(showlegend=False)
    return fig


def plot_sleep_vs_steps(features_df: pd.DataFrame) -> go.Figure:
    """Scatter: mean daily steps vs mean sleep hours, colored by BMI."""
    fig = px.scatter(
        features_df,
        x="mean_daily_steps",
        y="mean_sleep_hrs",
        color="bmi",
        color_continuous_scale="RdYlGn_r",
        labels={
            "mean_daily_steps": "Mean Daily Steps",
            "mean_sleep_hrs":   "Mean Sleep (hrs)",
            "bmi":              "BMI",
        },
        title="Activity vs Sleep Duration (colored by BMI)",
        opacity=0.6,
        trendline="ols",
    )
    return fig


def plot_sleep_consistency(features_df: pd.DataFrame) -> go.Figure:
    """Histogram of within-person sleep SD (consistency metric)."""
    fig = px.histogram(
        features_df,
        x="std_sleep_hrs",
        nbins=40,
        labels={"std_sleep_hrs": "SD of Nightly Sleep (hrs)"},
        title="Sleep Consistency Across Participants\n(lower SD = more consistent)",
        color_discrete_sequence=[PALETTE[1]],
    )
    return fig


def plot_sleep_over_time(sleep_df: pd.DataFrame, sample_n: int = 5) -> go.Figure:
    """Individual sleep timelines for a random sample of participants."""
    pids = sleep_df["person_id"].drop_duplicates().sample(min(sample_n, sleep_df["person_id"].nunique()), random_state=42)
    sample = sleep_df[sleep_df["person_id"].isin(pids)].copy()
    sample["sleep_date"] = pd.to_datetime(sample["sleep_date"])

    fig = px.line(
        sample,
        x="sleep_date",
        y="hours_asleep",
        color="person_id",
        labels={"hours_asleep": "Hours Asleep", "sleep_date": "Date", "person_id": "Participant"},
        title=f"Individual Sleep Timelines (n={sample_n} participants)",
    )
    fig.add_hline(y=7, line_dash="dash", line_color="gray")
    return fig


# ── Model result plots ────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing RMSE across models with error bars."""
    fig = px.bar(
        results_df,
        x="Model",
        y="rmse_mean",
        error_y="rmse_std",
        color="Model",
        labels={"rmse_mean": "RMSE (hrs)", "Model": ""},
        title="Model Comparison — Cross-Validated RMSE (lower is better)",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(showlegend=False)
    return fig


def plot_feature_importance(model, feature_names: list, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of feature importances (RF/GB models)."""
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.nlargest(top_n, "importance").sort_values("importance")

    fig = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": ""},
        title=f"Top {top_n} Feature Importances",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def plot_fairness(fairness_df: pd.DataFrame, subgroup_col: str) -> go.Figure:
    """Bar chart of per-subgroup RMSE with flagging."""
    colors = ["red" if f else PALETTE[0] for f in fairness_df["flagged"]]
    fig = go.Figure(go.Bar(
        x=fairness_df["subgroup"].astype(str),
        y=fairness_df["rmse"],
        marker_color=colors,
        text=fairness_df["n"].apply(lambda n: f"n={n}"),
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Model RMSE by {subgroup_col.title()} (red = potential disparity)",
        xaxis_title=subgroup_col.title(),
        yaxis_title="RMSE (hrs)",
    )
    return fig
