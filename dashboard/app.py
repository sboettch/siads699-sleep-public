"""
SIADS 699 — Sleep & Lifestyle in the All of Us Cohort
Streamlit Dashboard  |  Team Sleep Deprived  |  2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

st.set_page_config(
    page_title="Sleep & Lifestyle — All of Us",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG    = '#F8F9FA'
WHITE = '#FFFFFF'
DGRAY = '#1a1a2e'
MGRAY = '#6c757d'
LGRAY = '#dee2e6'
RED   = '#E05C3A'
BLUE  = '#3B7EC8'
GREEN = '#5AA469'
AMBER = '#E8A838'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #1a1a2e 0%, #16213e 100%) !important;
}
[data-testid="stSidebar"] * { color: #e8eaf6 !important; }
[data-testid="stSidebar"] .stRadio label { color: #e8eaf6 !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* Metric cards */
.kpi-row { display: flex; gap: 14px; margin: 16px 0; }
.kpi-card {
    flex: 1; background: white; border-radius: 14px;
    padding: 22px 18px; text-align: center;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    border-top: 4px solid #3B7EC8;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 18px rgba(59,126,200,0.18); }
.kpi-val   { font-size: 2rem; font-weight: 700; color: #3B7EC8; margin: 0; line-height: 1; }
.kpi-label { font-size: 0.78rem; font-weight: 600; color: #6c757d; margin: 6px 0 2px;
             text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-sub   { font-size: 0.75rem; color: #adb5bd; margin: 0; }

/* Callout boxes */
.box {
    border-radius: 10px; padding: 14px 18px; margin: 10px 0;
    font-size: 0.92rem; line-height: 1.55; color: #1a1a2e;
}
.box-blue  { background: #EEF4FF; border-left: 4px solid #3B7EC8; }
.box-red   { background: #FFF3EE; border-left: 4px solid #E05C3A; }
.box-green { background: #EDFFF4; border-left: 4px solid #5AA469; }
.box b     { color: #1a1a2e; }

/* Cluster cards */
.cl-grid { display: flex; gap: 14px; margin: 16px 0; }
.cl-card {
    flex: 1; background: white; border-radius: 14px; padding: 18px 16px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    transition: transform 0.15s, box-shadow 0.15s;
}
.cl-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.11); }
.cl-icon  { font-size: 1.6rem; margin-bottom: 6px; }
.cl-name  { font-weight: 700; font-size: 0.88rem; margin: 6px 0 4px; }
.cl-n     { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; margin: 0; }
.cl-pct   { font-size: 0.78rem; color: #adb5bd; margin: 0 0 10px; }
.cl-divider { border: none; border-top: 1px solid #f0f0f0; margin: 10px 0; }
.cl-stat  { font-size: 0.82rem; color: #495057; margin: 3px 0; }

/* Section header */
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #6c757d; margin: 24px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(text, kind='blue'):
    st.markdown(f'<div class="box box-{kind}">{text}</div>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def chart_fig(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=WHITE)
    ax.set_facecolor(WHITE)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(LGRAY); ax.spines['bottom'].set_color(LGRAY)
    ax.tick_params(colors=MGRAY, labelsize=9.5)
    ax.yaxis.grid(True, color=LGRAY, lw=0.8, zorder=0); ax.set_axisbelow(True)
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
    page = st.radio("", [
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
      <div class="kpi-card"><p class="kpi-val">59,757</p><p class="kpi-label">Participants</p><p class="kpi-sub">Fitbit wearers, AoU CDR v9</p></div>
      <div class="kpi-card"><p class="kpi-val">6.79 hrs</p><p class="kpi-label">Avg Sleep Duration</p><p class="kpi-sub">Below 7h recommendation</p></div>
      <div class="kpi-card"><p class="kpi-val">30%</p><p class="kpi-label">Short Sleep Nights</p><p class="kpi-sub">< 6 hrs avg per person</p></div>
      <div class="kpi-card"><p class="kpi-val">R²=0.184</p><p class="kpi-label">Best Consistency Model</p><p class="kpi-sub">HistGBM, Phase 2</p></div>
      <div class="kpi-card"><p class="kpi-val">4</p><p class="kpi-label">Sleep Phenotypes</p><p class="kpi-sub">Identified via KMeans</p></div>
    </div>""", unsafe_allow_html=True)

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

    target = st.radio("Prediction target:", ["Sleep Duration", "Sleep Consistency (IQR)"], horizontal=True)
    tkey  = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    tlbl  = "Sleep Duration" if "Duration" in target else "Sleep Consistency"

    p1 = cv1[cv1.target == tkey][['model','R2','MAE']].copy()
    p2 = cv2[cv2.target == tkey][['model','R2','MAE']].copy()

    col1, col2 = st.columns([3, 2])
    with col1:
        all_m = list(p1['model'].unique()) + [m for m in p2['model'].unique() if m not in p1['model'].values]
        x = np.arange(len(all_m)); w = 0.35
        p1d = p1.set_index('model')['R2'].reindex(all_m).fillna(0)
        p2d = p2.set_index('model')['R2'].reindex(all_m).fillna(0)

        fig, ax = chart_fig(10, 5)
        b1 = ax.bar(x - w/2, p1d, w, label='Phase 1 (baseline)', color=BLUE, alpha=0.6, edgecolor=WHITE)
        b2 = ax.bar(x + w/2, p2d, w, label='Phase 2 (+ HR/SES/survey)', color=GREEN, alpha=0.9, edgecolor=WHITE)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8, color=DGRAY)
        ax.set_xticks(x); ax.set_xticklabels(all_m, fontsize=10)
        ax.set_ylabel('R² (cross-validated)', fontsize=10, color=MGRAY)
        ax.set_title(f'Model Accuracy — {tlbl}', fontsize=12, fontweight='bold', color=DGRAY, pad=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(p1d.max(), p2d.max()) * 1.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        section("Phase 2 Results")
        d = p2.sort_values('R2', ascending=False).copy()
        d['R²'] = d['R2'].round(3); d['MAE (hrs)'] = d['MAE'].round(3)
        st.dataframe(d[['model','R²','MAE (hrs)']], hide_index=True, use_container_width=True)
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
    cards_html = '<div class="cl-grid">'
    for _, row in cl.sort_values('mean_sleep', ascending=False).iterrows():
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

    target = st.radio("Target:", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    tkey = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    sub  = fair[fair.target == tkey].copy()
    overall = sub['R2'].mean()
    sub['delta'] = sub['R2'] - overall
    sub = sub.sort_values('R2', ascending=False)

    col1, col2 = st.columns([3, 2])
    with col1:
        colors = [RED if d < -0.005 else GREEN if d > 0.005 else LGRAY for d in sub['delta']]
        fig, ax = chart_fig(9, 6)
        bars = ax.barh(sub['subgroup'], sub['R2'], color=colors, alpha=0.88, edgecolor=WHITE, height=0.55)
        ax.axvline(overall, color=DGRAY, lw=1.8, ls='--', alpha=0.6)
        ax.text(overall + 0.001, ax.get_ylim()[1]*0.97, f'avg {overall:.3f}', fontsize=8.5, color=DGRAY)
        for bar, val in zip(bars, sub['R2']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=9.5, color=DGRAY)
        ax.set_xlabel('R²', fontsize=10, color=MGRAY)
        ax.set_title(f'Model R² by Subgroup — {target}', fontsize=12, fontweight='bold', color=DGRAY, pad=10)
        ax.xaxis.grid(True, color=LGRAY, lw=0.8); ax.yaxis.grid(False)
        ax.legend(handles=[mpatches.Patch(color=RED,label='Below avg (gap)'),
                            mpatches.Patch(color=GREEN,label='Above avg')], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        section("Results by Subgroup")
        d = sub[['subgroup','N','R2']].copy()
        d['R²'] = d['R2'].round(3)
        d['vs avg'] = d['R2'].apply(lambda x: f"+{x-overall:.3f}" if x > overall else f"{x-overall:.3f}")
        st.dataframe(d[['subgroup','N','R²','vs avg']], hide_index=True, use_container_width=True)

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

    target = st.radio("Target:", ["Sleep Duration", "Sleep Consistency"], horizontal=True)
    imp = clean_imp(imp_d if "Duration" in target else imp_c)
    imp = imp[imp > 0.005]
    tlbl = "Sleep Duration" if "Duration" in target else "Sleep Consistency (IQR)"

    col1, col2 = st.columns([3, 2])
    with col1:
        simp = imp.sort_values()
        colors = [RED if i == simp.index[-1] else AMBER if i == simp.index[-2] else BLUE
                  for i in simp.index]
        fig, ax = chart_fig(9, max(5, len(simp)*0.5))
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
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        section("Top 5 Features")
        t5 = imp.head(5).reset_index(); t5.columns = ['Feature','Importance']
        t5['Importance'] = t5['Importance'].round(3)
        st.dataframe(t5, hide_index=True, use_container_width=True)
        st.divider()

        if "Duration" in target:
            box("🔑 <b>BMI</b> (0.162) and <b>female gender</b> (0.150) dominate — biological factors drive sleep duration more than behavioral ones in this cohort.", "blue")
            box("📌 The <b>BMI × steps</b> interaction (0.095) captures how the relationship between activity and sleep varies by body composition.", "blue")
        else:
            box("🔑 <b>Age × steps interaction</b> (0.442) is by far the dominant predictor — physical activity's effect on sleep regularity is strongly moderated by age.", "blue")
            box("📌 <b>Step variability</b> (0.109): people with inconsistent daily activity have inconsistent sleep.", "blue")
            box("📌 <b>Self-rated health</b> (0.097): perceived wellbeing is more tied to sleep regularity than to sleep duration.", "green")

    st.divider()
    section("Full Table")
    ft = imp.reset_index(); ft.columns = ['Feature','Importance']
    ft['Importance'] = ft['Importance'].round(4)
    st.dataframe(ft, hide_index=True, use_container_width=True)
