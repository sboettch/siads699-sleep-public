"""
SIADS 699 — Sleep & Lifestyle in the All of Us Cohort
Streamlit Dashboard  |  Team Sleep Deprived  |  2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from html import escape

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Sleep & Lifestyle — All of Us",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG    = '#0B1020'
WHITE = '#101827'
DGRAY = '#F8FAFC'
MGRAY = '#CBD5E1'
LGRAY = '#334155'
VGRAY = '#1E293B'
RED   = '#E05C3A'
BLUE  = '#3B7EC8'
GREEN = '#5AA469'
AMBER = '#E8A838'
PURPLE = '#7C5CC4'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top right, rgba(59,126,200,0.20), transparent 34rem),
        linear-gradient(180deg, #0B1020 0%, #111827 100%) !important;
}
[data-testid="stHeader"] {
    background: rgba(11,16,32,0.86) !important;
}
.block-container {
    max-width: 1240px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}
h1, h2, h3, p, li, label, span, div {
    letter-spacing: 0 !important;
}
h1 {
    color: #F8FAFC !important;
    font-weight: 750 !important;
    line-height: 1.08 !important;
}
p, li, label {
    color: #DDE6F3;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #16213e 100%) !important;
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e8eaf6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 6px 8px;
    margin: 2px 0;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.08);
}

/* Metric cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, minmax(142px, 1fr));
    gap: 14px;
    margin: 16px 0;
}
.kpi-card {
    background: rgba(16,24,39,0.92); border-radius: 8px;
    padding: 22px 18px; text-align: center;
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    border: 1px solid rgba(148,163,184,0.18);
    border-top: 4px solid var(--accent, #3B7EC8);
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 18px 38px rgba(59,126,200,0.20); }
.kpi-val   { font-size: clamp(1.45rem, 2.6vw, 2rem); font-weight: 750; color: var(--accent, #3B7EC8); margin: 0; line-height: 1; }
.kpi-label { font-size: 0.78rem; font-weight: 600; color: #CBD5E1; margin: 6px 0 2px;
             text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-sub   { font-size: 0.75rem; color: #94A3B8; margin: 0; }

/* Insight chips */
.insight-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(170px, 1fr));
    gap: 12px;
    margin: 16px 0 8px;
}
.insight {
    background: rgba(16,24,39,0.88);
    border: 1px solid rgba(148,163,184,0.18);
    border-left: 5px solid var(--accent, #3B7EC8);
    border-radius: 8px;
    padding: 14px 16px;
    min-height: 92px;
}
.insight-val {
    color: var(--accent, #3B7EC8);
    font-size: 1.45rem;
    font-weight: 750;
    line-height: 1;
    margin: 0 0 7px;
}
.insight-label {
    color: #F8FAFC;
    font-size: 0.9rem;
    font-weight: 650;
    margin: 0 0 4px;
}
.insight-sub {
    color: #CBD5E1;
    font-size: 0.78rem;
    margin: 0;
    line-height: 1.35;
}

/* Callout boxes */
.box {
    border-radius: 8px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.92rem; line-height: 1.55; color: #F8FAFC;
    border: 1px solid rgba(148,163,184,0.18);
}
.box-blue  { background: rgba(59,126,200,0.16); border-left: 4px solid #3B7EC8; }
.box-red   { background: rgba(224,92,58,0.16); border-left: 4px solid #E05C3A; }
.box-green { background: rgba(90,164,105,0.16); border-left: 4px solid #5AA469; }
.box b     { color: #FFFFFF; }

/* Cluster cards */
.cl-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(180px, 1fr));
    gap: 14px;
    margin: 16px 0;
}
.cl-card {
    background: rgba(16,24,39,0.92); border-radius: 8px; padding: 18px 16px;
    border: 1px solid rgba(148,163,184,0.18);
    box-shadow: 0 14px 30px rgba(0,0,0,0.22);
    transition: transform 0.15s, box-shadow 0.15s;
}
.cl-card:hover { transform: translateY(-2px); box-shadow: 0 18px 38px rgba(0,0,0,0.26); }
.cl-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.cl-name  { font-weight: 700; font-size: 0.88rem; margin: 6px 0 4px; }
.cl-n     { font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin: 0; }
.cl-pct   { font-size: 0.78rem; color: #94A3B8; margin: 0 0 10px; }
.cl-divider { border: none; border-top: 1px solid rgba(148,163,184,0.18); margin: 10px 0; }
.cl-stat  { font-size: 0.82rem; color: #CBD5E1; margin: 3px 0; }

/* Section header */
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94A3B8; margin: 24px 0 8px;
}

/* Compact tables rendered as responsive cards */
.table-card {
    background: rgba(16,24,39,0.92);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 8px;
    overflow-x: auto;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
}
.mini-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 360px;
}
.mini-table th {
    background: rgba(15,23,42,0.92);
    color: #CBD5E1;
    font-size: 0.73rem;
    text-transform: uppercase;
    text-align: left;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(148,163,184,0.18);
}
.mini-table td {
    color: #F8FAFC;
    font-size: 0.86rem;
    padding: 10px 12px;
    border-bottom: 1px solid rgba(148,163,184,0.12);
    white-space: nowrap;
}
.mini-table tr:last-child td {
    border-bottom: 0;
}
.rank-pill {
    display: inline-block;
    min-width: 28px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(59,126,200,0.22);
    color: #3B7EC8;
    font-weight: 700;
    text-align: center;
}

@media (max-width: 1100px) {
    .kpi-row { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
    .insight-grid { grid-template-columns: repeat(2, minmax(170px, 1fr)); }
    .cl-grid { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
}
@media (max-width: 760px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1.25rem;
    }
    h1 { font-size: 2rem !important; }
    .kpi-row, .insight-grid, .cl-grid {
        grid-template-columns: 1fr;
    }
    .kpi-card, .insight, .cl-card {
        padding: 16px 14px;
    }
    .kpi-card {
        text-align: left;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(text, kind='blue'):
    st.markdown(f'<div class="box box-{kind}">{text}</div>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def insight_cards(cards):
    html = '<div class="insight-grid">'
    for card in cards:
        html += f"""
        <div class="insight" style="--accent:{card['color']}">
            <p class="insight-val">{card['value']}</p>
            <p class="insight-label">{card['label']}</p>
            <p class="insight-sub">{card['sub']}</p>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def mini_table(df, formats=None, rank=False):
    formats = formats or {}
    headers = ''.join(f'<th>{escape(str(c))}</th>' for c in df.columns)
    rows = []
    for ridx, (_, row) in enumerate(df.iterrows(), start=1):
        cells = []
        for c in df.columns:
            val = row[c]
            if c in formats:
                val = formats[c](val)
            elif isinstance(val, float):
                val = f"{val:.3f}"
            cell = f'<span class="rank-pill">{ridx}</span>' if rank and c == df.columns[0] else escape(str(val))
            if rank and c == df.columns[0]:
                cell += f" {escape(str(row[c]))}"
            cells.append(f'<td>{cell}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<div class="table-card"><table class="mini-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
        unsafe_allow_html=True,
    )

def chart_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(LGRAY)
    ax.spines['bottom'].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=9.5)
    ax.yaxis.grid(True, color=LGRAY, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax

# ── Data ──────────────────────────────────────────────────────────────────────
DATA = os.path.dirname(os.path.abspath(__file__)) + "/"

@st.cache_data
def load():
    cv1  = pd.read_csv(DATA + "02_cv_results.csv")
    cv2  = pd.read_csv(DATA + "02_cv_results_v2.csv")
    fair = pd.read_csv(DATA + "02_fairness.csv")
    id_  = pd.read_csv(DATA + "02_importances_mean_sleep_hrs.csv", index_col=0, header=0)
    ic_  = pd.read_csv(DATA + "02_importances_iqr_sleep_hrs.csv",  index_col=0, header=0)
    cl   = pd.read_csv(DATA + "03_cluster_profiles.csv")
    return cv1, cv2, fair, id_, ic_, cl

cv1, cv2, fair, imp_d, imp_c, cl = load()

FEAT = {
    'n_nights':'Nights tracked','bmi':'BMI','female':'Female gender',
    'bmi_x_steps':'BMI × steps','age_x_steps':'Age × steps','age':'Age',
    'hr_missing':'HR missing','deprivation_index':'SES deprivation',
    'std_daily_steps':'Step variability','mean_resting_hr':'Resting heart rate',
    'mean_daily_steps':'Daily steps','self_rated_health':'Self-rated health',
    'pct_active_days':'% Active days','nonbinary':'Non-binary gender',
    'ses_missing':'SES missing','steps_missing':'Steps missing',
    'current_smoker':'Current smoker','drinks_per_week':'Drinks/week'
}

def clean_imp(df):
    s = df.iloc[:,0].copy(); s.index = [FEAT.get(i, i) for i in s.index]
    return s.sort_values(ascending=False)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Sleep & Lifestyle")
    st.markdown("**All of Us Research Program**")
    st.markdown("SIADS 699 &nbsp;·&nbsp; Team Sleep Deprived &nbsp;·&nbsp; 2026", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Dashboard section", [
        "📊  Overview",
        "🤖  Models",
        "👥  Sleep Phenotypes",
        "⚖️  Fairness",
        "🔍  Feature Importance",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Cohort:** 59,757 participants  \n**Source:** All of Us CDR v9  \n**Device:** Fitbit")
    st.divider()
    st.caption("🔒 Aggregate statistics only.\nNo individual-level data exported from Workbench.")

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Sleep & Lifestyle in the All of Us Cohort")
    st.markdown("*Can lifestyle and demographic factors predict sleep duration and consistency in 59,757 Fitbit wearers?*")
    st.divider()

    st.markdown("""
    <div class="kpi-row">
      <div class="kpi-card" style="--accent:#3B7EC8"><p class="kpi-val">59,757</p><p class="kpi-label">Participants</p><p class="kpi-sub">Fitbit wearers, AoU CDR v9</p></div>
      <div class="kpi-card" style="--accent:#E8A838"><p class="kpi-val">6.79 hrs</p><p class="kpi-label">Avg Sleep Duration</p><p class="kpi-sub">Below 7h recommendation</p></div>
      <div class="kpi-card" style="--accent:#E05C3A"><p class="kpi-val">30%</p><p class="kpi-label">Short Sleep Nights</p><p class="kpi-sub">< 6 hrs avg per person</p></div>
      <div class="kpi-card" style="--accent:#5AA469"><p class="kpi-val">R²=0.184</p><p class="kpi-label">Best Consistency Model</p><p class="kpi-sub">HistGBM, Phase 2</p></div>
      <div class="kpi-card" style="--accent:#7C5CC4"><p class="kpi-val">4</p><p class="kpi-label">Sleep Phenotypes</p><p class="kpi-sub">Identified via KMeans</p></div>
    </div>""", unsafe_allow_html=True)

    insight_cards([
        {"value": "+16%", "label": "Model lift", "sub": "Best Phase 2 model vs. Phase 1 baseline", "color": GREEN},
        {"value": "44%", "label": "Dominant signal", "sub": "Age × steps share of consistency importance", "color": BLUE},
        {"value": "40%", "label": "Equity gap", "sub": "Lower duration R² for UBR vs. White participants", "color": RED},
        {"value": "42%", "label": "Largest phenotype", "sub": "Consistent Good Sleepers", "color": PURPLE},
    ])

    st.markdown(" ")
    col1, col2 = st.columns(2)

    with col1:
        section("Key Findings")
        box("📈 <b>HistGBM achieves R²=0.099 (duration) and R²=0.184 (consistency)</b> — a +16% improvement over baseline from adding resting HR, SES, and self-rated health.", "blue")
        box("🔑 <b>Age × daily steps interaction explains 44% of variance in sleep consistency</b> — the single dominant predictor across all models.", "blue")
        box("👥 <b>Four sleep phenotypes identified:</b> 42% Consistent Good Sleepers, 24% Chronic Short & Variable, 24% Short but Regular, 9% Variable Long Sleepers.", "green")
        box("⚖️ <b>40% fairness gap:</b> Model accuracy for UBR participants is substantially lower than for White participants — a structural limitation requiring disclosure.", "red")
        box("💤 <b>Sleep consistency is more predictable than duration</b> across all models — lifestyle shapes regularity more than length.", "blue")

    with col2:
        section("Research Questions")
        st.markdown("""
**RQ1** Can lifestyle + demographic factors predict sleep duration and consistency?

**RQ2** Which features are most predictive?

**RQ3** Are there distinct sleep phenotypes in the AoU cohort?

**RQ4** Is model performance equitable across demographic groups?
        """)
        st.divider()
        section("Study Design")
        st.markdown("""
| | |
|---|---|
| **Data** | AoU CDR v9 Fitbit sleep + activity |
| **Inclusion** | ≥4 nights, 4–12 hrs range |
| **Features** | Sleep, steps, demographics, BMI, HR, SES, survey |
| **Validation** | 5-fold participant-level CV |
| **Clustering** | KMeans k=4 |
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Models" in page:
    st.title("Model Performance")
    st.markdown("5-fold cross-validated R² and MAE across all 59,757 participants.")
    st.divider()

    ctrl1, ctrl2 = st.columns([1, 1])
    with ctrl1:
        target = st.radio("Prediction target:", ["Sleep Duration", "Sleep Consistency (IQR)"], horizontal=True)
    with ctrl2:
        model_metric = st.radio("Chart metric:", ["R² lift", "MAE change"], horizontal=True)
    tkey  = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    tlbl  = "Sleep Duration" if "Duration" in target else "Sleep Consistency"

    p1 = cv1[cv1.target == tkey][['model','R2','MAE']].copy()
    p2 = cv2[cv2.target == tkey][['model','R2','MAE']].copy()

    all_m = list(p1['model'].unique()) + [m for m in p2['model'].unique() if m not in p1['model'].values]
    metric_col = 'R2' if model_metric == "R² lift" else 'MAE'
    metric_label = 'R² (higher is better)' if metric_col == 'R2' else 'MAE hours (lower is better)'
    p1d = p1.set_index('model')[metric_col].reindex(all_m).fillna(0)
    p2d = p2.set_index('model')[metric_col].reindex(all_m).fillna(0)

    y = np.arange(len(all_m))
    best_model = p2d.idxmax() if metric_col == 'R2' else p2d.idxmin()
    fig, ax = chart_fig(11.5, 5.2)
    ax.hlines(y, p1d, p2d, color=LGRAY, lw=9, zorder=1)
    ax.scatter(p1d, y, s=145, color=BLUE, alpha=0.45, edgecolor=WHITE, linewidth=1.2, label='Phase 1')
    ax.scatter(p2d, y, s=[230 if m == best_model else 165 for m in all_m],
               color=[GREEN if m == best_model else AMBER for m in all_m],
               alpha=0.95, edgecolor=WHITE, linewidth=1.4, label='Phase 2')
    for i, m in enumerate(all_m):
        delta = p2d[m] - p1d[m]
        label_delta = delta if metric_col == 'R2' else -delta
        ax.text(max(p1d[m], p2d[m]) + (0.006 if metric_col == 'R2' else 0.018), i,
                f"{p2d[m]:.3f}  ({label_delta:+.3f})",
                va='center', fontsize=9.5, color=DGRAY, fontweight='700' if m == best_model else '500')
    ax.set_yticks(y)
    ax.set_yticklabels(all_m, fontsize=10, color=DGRAY)
    ax.set_xlabel(metric_label, fontsize=10, color=MGRAY)
    ax.set_title(f'Model {model_metric} — {tlbl}', fontsize=12, fontweight='bold', color=DGRAY, pad=12)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8)
    ax.yaxis.grid(False)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(p1d.max(), p2d.max()) * (1.35 if metric_col == 'R2' else 1.18))
    for sp in ['left']:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    section("Phase 2 Results")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        d = p2.sort_values('R2', ascending=False).copy()
        d['R²'] = d['R2'].round(3); d['MAE (hrs)'] = d['MAE'].round(3)
        mini_table(d[['model','R²','MAE (hrs)']], rank=True)
    with detail2:
        bm = d.iloc[0]
        box(f"Best: <b>{bm['model']}</b><br>R² = {bm['R2']:.3f} &nbsp;|&nbsp; MAE = {bm['MAE']*60:.0f} min", "green")
        box("Low R² is expected — sleep is driven by unmeasured factors: genetics, stress, medications. 10–18% explained variance aligns with the sleep epidemiology literature.", "blue")

    st.divider()
    section("Phase 1 → Phase 2 Improvement")
    c1, c2 = st.columns(2)
    for col, (tgt, lbl) in zip([c1, c2], [("mean_sleep_hrs","Duration"), ("iqr_sleep_hrs","Consistency")]):
        b1v = cv1[cv1.target == tgt]['R2'].max()
        b2v = cv2[cv2.target == tgt]['R2'].max()
        col.metric(f"{lbl} R²", f"{b2v:.3f}", f"+{(b2v-b1v)/b1v*100:.0f}% vs Phase 1")

# ═══════════════════════════════════════════════════════════════════════════════
# SLEEP PHENOTYPES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Phenotypes" in page:
    st.title("Sleep Phenotypes")
    st.markdown("KMeans clustering (k=4) identifies four distinct sleep types based on duration, variability, and short sleep rate.")
    st.divider()

    CC = {'Consistent Good Sleepers': GREEN, 'Short but Regular': BLUE,
          'Chronic Short & Variable': RED,   'Variable Long Sleepers': AMBER}
    CI = {'Consistent Good Sleepers':'✅','Short but Regular':'⚠️',
          'Chronic Short & Variable':'❌','Variable Long Sleepers':'🔄'}

    total = cl['N'].sum()
    pcol1, pcol2, pcol3 = st.columns([1, 1, 1])
    with pcol1:
        sort_choice = st.selectbox("Sort phenotype cards by:", ["Sleep duration", "Cohort share", "Variability", "Short-night rate"])
    with pcol2:
        x_choice = st.selectbox("Map x-axis:", ["Sleep duration", "Daily steps", "Age", "BMI"])
    with pcol3:
        y_choice = st.selectbox("Map y-axis:", ["Variability", "Short-night rate", "Daily steps", "BMI"])

    sort_map = {
        "Sleep duration": ("mean_sleep", False),
        "Cohort share": ("N", False),
        "Variability": ("mean_iqr", False),
        "Short-night rate": ("pct_short", False),
    }
    x_map = {"Sleep duration": "mean_sleep", "Daily steps": "mean_steps", "Age": "mean_age", "BMI": "mean_bmi"}
    y_map = {"Variability": "mean_iqr", "Short-night rate": "pct_short", "Daily steps": "mean_steps", "BMI": "mean_bmi"}
    sort_col, ascending = sort_map[sort_choice]
    cards_html = '<div class="cl-grid">'
    for _, row in cl.sort_values(sort_col, ascending=ascending).iterrows():
        nm = row['cluster_label']; c = CC.get(nm, BLUE); ic = CI.get(nm, '•')
        cards_html += f"""
        <div class="cl-card" style="border-top: 5px solid {c}">
            <div class="cl-icon">{ic}</div>
            <div class="cl-name" style="color:{c}">{nm}</div>
            <p class="cl-n">{row['N']:,}</p>
            <p class="cl-pct">{row['N']/total*100:.0f}% of cohort</p>
            <hr class="cl-divider">
            <p class="cl-stat">😴 {row['mean_sleep']:.2f} hrs avg</p>
            <p class="cl-stat">📊 IQR {row['mean_iqr']:.2f} hrs variability</p>
            <p class="cl-stat">🌙 {row['pct_short']:.0%} short nights</p>
            <p class="cl-stat">👟 {row['mean_steps']:,.0f} steps/day</p>
            <p class="cl-stat">📅 Age {row['mean_age']:.0f} · BMI {row['mean_bmi']:.1f}</p>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    section("Phenotype Map")
    fig, ax = chart_fig(10, 5)
    for _, row in cl.iterrows():
        nm = row['cluster_label']
        c = CC.get(nm, BLUE)
        size = 240 + (row['N'] / total) * 2300
        xval = row[x_map[x_choice]]
        yval = row[y_map[y_choice]]
        ax.scatter(xval, yval, s=size, color=c, alpha=0.82,
                   edgecolor=WHITE, linewidth=2.2, zorder=3)
        ax.text(xval, yval, f"{row['N']/total:.0%}",
                ha='center', va='center', fontsize=11, color='white', fontweight='750', zorder=4)
        ax.annotate(nm, (xval, yval),
                    xytext=(10, 6), textcoords='offset points', fontsize=9.5,
                    color=DGRAY, fontweight='650')
    if x_choice == "Sleep duration":
        ax.axvline(7, color=AMBER, ls='--', lw=1.6, alpha=0.8)
        ax.text(7.03, ax.get_ylim()[0] + 0.04, '7h reference', color=AMBER, fontsize=8.8, fontweight='650')
    ax.set_xlabel(x_choice, fontsize=10, color=MGRAY)
    ax.set_ylabel(y_choice, fontsize=10, color=MGRAY)
    ax.set_title('Sleep phenotypes by duration, variability, and cohort share', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    section("Feature Heatmap")
    fc  = ['mean_sleep','mean_iqr','pct_short','mean_steps','mean_bmi','mean_age']
    fl  = ['Sleep Duration','Variability','Short Night %','Daily Steps','BMI','Age']
    z   = cl.set_index('cluster_label')[fc].copy()
    zn  = (z - z.mean()) / z.std()
    ORD = ['Consistent Good Sleepers','Short but Regular','Chronic Short & Variable','Variable Long Sleepers']
    zn  = zn.reindex([r for r in ORD if r in zn.index])

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor=WHITE); ax.set_facecolor(WHITE)
    im = ax.imshow(zn.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(fl))); ax.set_xticklabels(fl, fontsize=10, color=DGRAY)
    ax.set_yticks(range(len(zn))); ax.set_yticklabels(zn.index, fontsize=10, color=DGRAY)
    ax.tick_params(left=False, bottom=False)
    for i in range(len(zn)):
        for j, feat in enumerate(fc):
            raw = z.loc[zn.index[i], feat]; zv = zn.values[i, j]
            fmt = f'{raw:.0%}' if feat=='pct_short' else (f'{raw:,.0f}' if feat=='mean_steps' else f'{raw:.1f}')
            ax.text(j, i, fmt, ha='center', va='center', fontsize=9.5,
                    color='white' if abs(zv)>1.2 else DGRAY, fontweight='600')
    plt.colorbar(im, ax=ax, label='z-score', fraction=0.02, pad=0.02)
    ax.set_title('Cluster Profiles — red = above average, blue = below average', fontsize=11, fontweight='bold', color=DGRAY, pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    box("⚠️ <b>Short but Regular</b> (24%): 55% of nights average under 6 hours — severe chronic deprivation with a consistent schedule. High steps suggest trading sleep for activity.", "red")
    box("💡 <b>Variable Long Sleepers</b> (9%): Highest sleep duration but lowest step count and extreme night-to-night variability. Likely shift workers, retirees, or those with irregular schedules.", "blue")

# ═══════════════════════════════════════════════════════════════════════════════
# FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:
    st.title("Model Fairness")
    st.markdown("Does the model perform equally across demographic groups?")
    st.divider()

    fcol1, fcol2 = st.columns([1, 1])
    with fcol1:
        target = st.radio("Target:", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    with fcol2:
        fairness_view = st.radio("Chart view:", ["Deviation from average", "Raw R²"], horizontal=True)
    tkey = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    sub  = fair[fair.target == tkey].copy()
    overall = sub['R2'].mean()
    sub['delta'] = sub['R2'] - overall
    sort_metric = 'delta' if fairness_view == "Deviation from average" else 'R2'
    sub = sub.sort_values(sort_metric, ascending=False)

    colors = [RED if d < -0.005 else GREEN if d > 0.005 else LGRAY for d in sub['delta']]
    fig, ax = chart_fig(11.5, 6)
    plot_values = sub['delta'] if fairness_view == "Deviation from average" else sub['R2']
    bars = ax.barh(sub['subgroup'], plot_values, color=colors, alpha=0.9, edgecolor=WHITE, height=0.58)
    if fairness_view == "Deviation from average":
        ax.axvline(0, color=DGRAY, lw=1.8, alpha=0.65)
    else:
        ax.axvline(overall, color=AMBER, lw=1.8, ls='--', alpha=0.85)
    for bar, val, delta, r2 in zip(bars, plot_values, sub['delta'], sub['R2']):
        ha = 'left' if val >= 0 else 'right'
        offset = 0.002 if val >= 0 else -0.002
        label = f"{delta:+.3f}  R² {r2:.3f}" if fairness_view == "Deviation from average" else f"R² {r2:.3f}  ({delta:+.3f})"
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, label,
                va='center', ha=ha, fontsize=9.3, color=DGRAY)
    ax.set_xlabel('R² difference from subgroup average' if fairness_view == "Deviation from average" else "R²", fontsize=10, color=MGRAY)
    ax.set_title(f'Fairness {fairness_view} — {target}', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8); ax.yaxis.grid(False)
    ax.legend(handles=[mpatches.Patch(color=RED,label='Below avg (gap)'),
                        mpatches.Patch(color=GREEN,label='Above avg')], fontsize=9)
    if fairness_view == "Deviation from average":
        lim = max(abs(sub['delta'].min()), abs(sub['delta'].max())) * 1.35
        ax.set_xlim(-lim, lim)
    else:
        ax.set_xlim(0, sub['R2'].max() * 1.25)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    section("Results by Subgroup")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        d = sub[['subgroup','N','R2']].copy()
        d['R²'] = d['R2'].round(3)
        d['vs avg'] = d['R2'].apply(lambda x: f"+{x-overall:.3f}" if x > overall else f"{x-overall:.3f}")
        mini_table(d[['subgroup','N','R²','vs avg']], formats={'N': lambda x: f"{int(x):,}"})

    with detail2:
        ubr = sub[sub.subgroup=='UBR']['R2'].values[0]
        wht = sub[sub.subgroup=='White']['R2'].values[0]
        gap = (wht - ubr) / wht * 100
        young = sub[sub.subgroup=='Age 18-40']['R2'].values[0]

        box(f"⚠️ <b>{gap:.0f}% fairness gap:</b> UBR R²={ubr:.3f} vs White R²={wht:.3f}. The model explains substantially less for underrepresented minority participants — driven by training data composition (70% White cohort).", "red")
        box(f"📌 <b>18–40 year-olds</b> have the highest R²={young:.3f}. Sleep is most behaviorally driven in younger adults — lifestyle factors are stronger predictors.", "blue")

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Importance" in page:
    st.title("Feature Importance")
    st.markdown("Random Forest feature importances trained on the full cohort (mean decrease in impurity).")
    st.divider()

    icol1, icol2, icol3 = st.columns([1, 1, 1])
    with icol1:
        target = st.radio("Target:", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    with icol2:
        top_n = st.slider("Features shown:", 5, 18, 12)
    with icol3:
        show_table = st.toggle("Show full table", value=True)
    imp = clean_imp(imp_d if "Duration" in target else imp_c)
    imp = imp[imp > 0.005]
    imp_chart = imp.head(top_n)
    tlbl = "Sleep Duration" if "Duration" in target else "Sleep Consistency (IQR)"

    simp = imp_chart.sort_values()
    colors = []
    for i in simp.index:
        if i == simp.index[-1]:
            colors.append(RED)
        elif i in list(simp.index[-3:-1]):
            colors.append(AMBER)
        else:
            colors.append(BLUE)
    fig, ax = chart_fig(11.5, max(5.4, len(simp)*0.46))
    bars = ax.barh(simp.index, simp.values, color=colors, alpha=0.88, edgecolor=WHITE, height=0.58)
    for bar, val in zip(bars, simp.values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9, color=DGRAY)
    ax.set_xlabel('Feature Importance (RF)', fontsize=10, color=MGRAY)
    ax.set_title(f'Feature Importances — {tlbl}', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
    ax.xaxis.grid(True, color=LGRAY, lw=0.8); ax.yaxis.grid(False)
    ax.tick_params(left=False)
    for sp in ['left']: ax.spines[sp].set_visible(False)
    ax.set_xlim(0, simp.max() * 1.25)
    ax.text(simp.max() * 1.23, len(simp) - 1, 'top driver', ha='right', va='center',
            fontsize=8.8, color=RED, fontweight='700')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    section("Top 5 Features")
    detail1, detail2 = st.columns([1, 1])
    with detail1:
        t5 = imp.head(5).reset_index(); t5.columns = ['Feature','Importance']
        t5['Importance'] = t5['Importance'].round(3)
        mini_table(t5, rank=True)

    with detail2:
        top3_share = imp.head(3).sum() / imp.sum() * 100
        insight_cards([
            {"value": f"{top3_share:.0f}%", "label": "Top-3 concentration", "sub": "Share of displayed importance in the first three signals", "color": RED if top3_share > 50 else BLUE},
            {"value": f"{imp.iloc[0]/imp.iloc[1]:.1f}×", "label": "Lead feature ratio", "sub": "Top feature compared with the second-ranked feature", "color": AMBER},
        ])

        if "Duration" in target:
            box("🔑 <b>BMI</b> (0.162) and <b>female gender</b> (0.150) dominate — biological factors drive sleep duration more than behavioral ones in this cohort.", "blue")
            box("📌 The <b>BMI × steps</b> interaction (0.095) captures how the relationship between activity and sleep varies by body composition.", "blue")
        else:
            box("🔑 <b>Age × steps interaction</b> (0.442) is by far the dominant predictor — physical activity's effect on sleep regularity is strongly moderated by age.", "blue")
            box("📌 <b>Step variability</b> (0.109): people with inconsistent daily activity have inconsistent sleep.", "blue")
            box("📌 <b>Self-rated health</b> (0.097): perceived wellbeing is more tied to sleep regularity than to sleep duration.", "green")

    if show_table:
        st.divider()
        section("Full Table")
        ft = imp.reset_index(); ft.columns = ['Feature','Importance']
        ft['Importance'] = ft['Importance'].round(4)
        mini_table(ft)
