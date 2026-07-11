"""
app.py — SIADS 699 Sleep Health Dashboard
------------------------------------------
Run locally:  streamlit run dashboard/app.py
In Workbench: streamlit run dashboard/app.py (after exporting aggregate CSVs)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.features import clean_sleep, clean_steps, make_participant_features, add_targets
from src.viz import (
    plot_sleep_distribution, plot_sleep_by_group, plot_sleep_vs_steps,
    plot_sleep_consistency, plot_sleep_over_time,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep Health Dashboard — Team Sleep Deprived",
    page_icon="😴",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), "..", "synthetic_data")
    sleep_raw = pd.read_csv(os.path.join(base, "synthetic_sleep.csv"))
    steps_raw = pd.read_csv(os.path.join(base, "synthetic_steps.csv"))
    demo_raw  = pd.read_csv(os.path.join(base, "synthetic_demographics.csv"))

    sleep_clean = clean_sleep(sleep_raw)
    steps_clean = clean_steps(steps_raw)
    features    = make_participant_features(sleep_clean, steps_clean, demo_raw)
    features    = add_targets(features)

    return sleep_clean, steps_clean, features

sleep_df, steps_df, features_df = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("😴 Sleep Health Explorer")
st.sidebar.markdown("*SIADS 699 — Team Sleep Deprived*")
st.sidebar.markdown("---")

age_range = st.sidebar.slider(
    "Age range", int(features_df.age.min()), int(features_df.age.max()),
    (18, 80)
)
bmi_range = st.sidebar.slider(
    "BMI range", float(features_df.bmi.min()), float(features_df.bmi.max()),
    (15.0, 65.0)
)
regions = st.sidebar.multiselect(
    "Region", options=sorted(features_df.region.unique()),
    default=list(features_df.region.unique())
)

mask = (
    features_df.age.between(*age_range) &
    features_df.bmi.between(*bmi_range) &
    features_df.region.isin(regions)
)
filtered = features_df[mask]
filtered_sleep = sleep_df[sleep_df.person_id.isin(filtered.person_id)]

st.sidebar.markdown(f"**{len(filtered):,}** participants selected")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("😴 Sleep Health & Lifestyle Factors")
st.markdown(
    "Exploring associations between sleep duration/consistency and lifestyle "
    "factors using NIH All of Us wearables data. *Showing synthetic data for development.*"
)
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Participants",     f"{len(filtered):,}")
c2.metric("Median Sleep",     f"{filtered.median_sleep_hrs.median():.1f} hrs")
c3.metric("Avg Daily Steps",  f"{filtered.mean_daily_steps.mean():,.0f}")
c4.metric("Short Sleep (<6h)", f"{filtered.pct_short_sleep.mean():.1%}")

st.markdown("---")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Sleep Distributions",
    "🏃 Activity vs Sleep",
    "👥 Demographics",
    "📈 Individual Timelines",
])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_sleep_distribution(filtered_sleep), use_container_width=True)
    with col2:
        st.plotly_chart(plot_sleep_consistency(filtered), use_container_width=True)

with tab2:
    st.plotly_chart(plot_sleep_vs_steps(filtered), use_container_width=True)

with tab3:
    group = st.selectbox("Group by", ["race", "gender", "income_tier", "region"])
    st.plotly_chart(plot_sleep_by_group(filtered, group), use_container_width=True)

with tab4:
    n_sample = st.slider("Number of participants to show", 3, 20, 5)
    st.plotly_chart(plot_sleep_over_time(filtered_sleep, sample_n=n_sample), use_container_width=True)

st.markdown("---")
st.caption(
    "Data: NIH All of Us Research Program (Patten et al., Nat Med 2026). "
    "Dashboard shows synthetic data only — no participant-level data is stored or displayed."
)
