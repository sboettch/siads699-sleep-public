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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sleep & Lifestyle — All of Us",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette (538-inspired) ──────────────────────────────────────────────
BG    = '#F0F0F0'
DGRAY = '#3C3C3C'
GRAY  = '#D0D0D0'
RED   = '#E05C3A'
BLUE  = '#3B7EC8'
GREEN = '#5AA469'
AMBER = '#E8A838'

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
    border-left: 4px solid #3B7EC8;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #3B7EC8; margin: 0; }
.metric-label { font-size: 0.85rem; color: #666; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-sub   { font-size: 0.8rem; color: #999; margin: 2px 0 0 0; }
.cluster-card { background: white; border-radius: 12px; padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 12px; }
.finding-box { background: #EEF4FF; border-left: 4px solid #3B7EC8;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 12px 0; font-size: 0.95rem; }
.warning-box { background: #FFF3EE; border-left: 4px solid #E05C3A;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 12px 0; font-size: 0.95rem; }
section[data-testid="stSidebar"] { background: #1a1a2e; }
section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
DATA = os.path.dirname(os.path.abspath(__file__)) + "/"

@st.cache_data
def load_data():
    cv1   = pd.read_csv(DATA + "02_cv_results.csv")
    cv2   = pd.read_csv(DATA + "02_cv_results_v2.csv")
    fair  = pd.read_csv(DATA + "02_fairness.csv")
    imp_d = pd.read_csv(DATA + "02_importances_mean_sleep_hrs.csv", index_col=0, header=0)
    imp_c = pd.read_csv(DATA + "02_importances_iqr_sleep_hrs.csv",  index_col=0, header=0)
    cl    = pd.read_csv(DATA + "03_cluster_profiles.csv")
    return cv1, cv2, fair, imp_d, imp_c, cl

cv1, cv2, fair, imp_d, imp_c, cl = load_data()

FEAT_NAMES = {
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
    s = df.iloc[:,0].copy()
    s.index = [FEAT_NAMES.get(i, i) for i in s.index]
    return s.sort_values(ascending=False)

def fte_ax(ax):
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌙 Sleep & Lifestyle")
    st.markdown("**All of Us Research Program**")
    st.markdown("SIADS 699 · Team Sleep Deprived · 2026")
    st.divider()
    page = st.radio("Navigate", [
        "📊  Overview",
        "🤖  Models",
        "👥  Sleep Phenotypes",
        "⚖️  Fairness",
        "🔍  Feature Importance",
    ])
    st.divider()
    st.markdown("**Cohort:** 59,757 participants")
    st.markdown("**Source:** All of Us CDR v9")
    st.caption("Aggregate statistics only.\nNo individual-level data exported.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Sleep & Lifestyle in the All of Us Cohort")
    st.markdown("*Can lifestyle and demographic factors predict sleep duration and consistency in 59,757 Fitbit wearers?*")
    st.divider()

    for col, (val, label, sub) in zip(st.columns(5), [
        ("59,757","Participants","Fitbit wearers, AoU CDR v9"),
        ("6.79 hrs","Avg Sleep Duration","Below 7h recommendation"),
        ("30%","Short Sleep Nights","< 6 hrs avg per person"),
        ("R²=0.184","Best Consistency Model","HistGBM, Phase 2"),
        ("4","Sleep Phenotypes","Identified via KMeans"),
    ]):
        col.markdown(f'<div class="metric-card"><p class="metric-value">{val}</p>'
                     f'<p class="metric-label">{label}</p><p class="metric-sub">{sub}</p></div>',
                     unsafe_allow_html=True)

    st.markdown("### ")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Key Findings")
        for icon, text in [
            ("📈","HistGBM achieves R²=0.099 (duration) and R²=0.184 (consistency) — a +16% improvement over baseline."),
            ("🔑","Age × daily steps interaction explains 44% of variance in sleep consistency — the dominant predictor."),
            ("👥","Four sleep phenotypes identified: 42% Consistent Good Sleepers, 24% Chronic Short & Variable."),
            ("⚖️","Model accuracy is 40% lower for UBR participants vs White — a structural fairness gap."),
            ("💤","Sleep consistency is more predictable than duration across all models."),
        ]:
            st.markdown(f'<div class="finding-box">{icon} {text}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### Research Questions")
        st.markdown("""
**RQ1:** Can lifestyle + demographic factors predict sleep duration and consistency?

**RQ2:** Which features matter most?

**RQ3:** Are there distinct sleep phenotypes in the AoU cohort?

**RQ4:** Is model performance equitable across demographic groups?
        """)
        st.divider()
        st.markdown("### Design")
        st.markdown("""
- **Data:** AoU CDR v9 Fitbit sleep, activity, demographics, BMI, HR, SES, survey
- **Inclusion:** ≥4 nights, 4–12 hrs sleep range
- **Validation:** 5-fold participant-level CV
- **Note:** All reported metrics are aggregate-only
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Models" in page:
    st.title("Model Performance")
    st.markdown("5-fold cross-validated R² and MAE across all 59,757 participants.")
    st.divider()

    target = st.radio("Prediction target:", ["Sleep Duration", "Sleep Consistency (IQR)"], horizontal=True)
    tgt_key   = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    tgt_label = "Sleep Duration" if "Duration" in target else "Sleep Consistency"

    p1 = cv1[cv1.target==tgt_key][['model','R2','MAE']].copy()
    p2 = cv2[cv2.target==tgt_key][['model','R2','MAE']].copy()

    col1, col2 = st.columns([2,1])
    with col1:
        fig, ax = plt.subplots(figsize=(10,5), facecolor='white')
        ax.set_facecolor('white')
        all_models = list(p1['model'].unique()) + [m for m in p2['model'].unique() if m not in p1['model'].values]
        x = np.arange(len(all_models)); w = 0.35
        p1d = p1.set_index('model')['R2'].reindex(all_models).fillna(0)
        p2d = p2.set_index('model')['R2'].reindex(all_models).fillna(0)
        b1 = ax.bar(x-w/2, p1d, w, label='Phase 1 (baseline)', color=BLUE, alpha=0.7, edgecolor='white')
        b2 = ax.bar(x+w/2, p2d, w, label='Phase 2 (+ HR/SES/survey)', color=GREEN, alpha=0.85, edgecolor='white')
        for bar in list(b1)+list(b2):
            h = bar.get_height()
            if h > 0.005:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.002, f'{h:.3f}',
                        ha='center', va='bottom', fontsize=8.5, color=DGRAY)
        ax.set_xticks(x); ax.set_xticklabels(all_models, fontsize=11)
        ax.set_ylabel('R² (cross-validated)', fontsize=11)
        ax.set_title(f'Model R² — {tgt_label}', fontsize=13, fontweight='bold', pad=10)
        ax.legend(fontsize=10); fte_ax(ax)
        ax.set_ylim(0, max(p1d.max(), p2d.max()) * 1.35)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("### Phase 2 Results")
        disp = p2.sort_values('R2', ascending=False).copy()
        disp['R²'] = disp['R2'].round(3); disp['MAE (hrs)'] = disp['MAE'].round(3)
        st.dataframe(disp[['model','R²','MAE (hrs)']], hide_index=True, use_container_width=True)
        bm = disp.iloc[0]
        st.markdown(f'<div class="finding-box">Best: <b>{bm["model"]}</b><br>R² = {bm["R2"]:.3f} | MAE = {bm["MAE"]:.3f} hrs (~{bm["MAE"]*60:.0f} min)</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-box">Low R² is expected and defensible — sleep is driven by unmeasured factors: genetics, stress, medications. 10–18% explained variance is consistent with the sleep epidemiology literature.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Phase 1 → Phase 2 Improvement")
    c1, c2 = st.columns(2)
    for col, (tgt, lbl) in zip([c1,c2],[("mean_sleep_hrs","Duration"),("iqr_sleep_hrs","Consistency")]):
        b1 = cv1[cv1.target==tgt]['R2'].max()
        b2 = cv2[cv2.target==tgt]['R2'].max()
        col.metric(f"{lbl} R²", f"{b2:.3f}", f"+{(b2-b1)/b1*100:.0f}% vs Phase 1")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SLEEP PHENOTYPES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Phenotypes" in page:
    st.title("Sleep Phenotypes")
    st.markdown("KMeans clustering (k=4) on sleep duration, variability, and short sleep rate identifies four distinct types.")
    st.divider()

    CC = {'Consistent Good Sleepers':GREEN,'Short but Regular':BLUE,
          'Chronic Short & Variable':RED,'Variable Long Sleepers':AMBER}
    CI = {'Consistent Good Sleepers':'✅','Short but Regular':'⚠️',
          'Chronic Short & Variable':'❌','Variable Long Sleepers':'🔄'}

    total = cl['N'].sum()
    for col, (_, row) in zip(st.columns(4), cl.sort_values('mean_sleep', ascending=False).iterrows()):
        nm = row['cluster_label']; c = CC.get(nm, BLUE); ic = CI.get(nm, '•')
        col.markdown(f"""
        <div class="cluster-card" style="border-left:5px solid {c}">
            <div style="font-size:1.5rem">{ic}</div>
            <div style="font-weight:700;font-size:0.9rem;color:{c};margin:8px 0 4px">{nm}</div>
            <div style="font-size:1.7rem;font-weight:700;color:{DGRAY}">{row['N']:,}</div>
            <div style="font-size:0.8rem;color:#999">{row['N']/total*100:.0f}% of cohort</div>
            <hr style="margin:10px 0;border:none;border-top:1px solid #eee">
            <div style="font-size:0.85rem;color:{DGRAY}">
            😴 {row['mean_sleep']:.2f} hrs<br>
            📊 IQR {row['mean_iqr']:.2f} hrs<br>
            🌙 {row['pct_short']:.0%} short nights<br>
            👟 {row['mean_steps']:,.0f} steps/day<br>
            📅 Age {row['mean_age']:.0f} · BMI {row['mean_bmi']:.1f}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### Feature Heatmap")
    fc = ['mean_sleep','mean_iqr','pct_short','mean_steps','mean_bmi','mean_age']
    fl = ['Sleep Duration','Variability','Short Night %','Daily Steps','BMI','Age']
    z = cl.set_index('cluster_label')[fc].copy()
    zn = (z - z.mean()) / z.std()
    ORDER = ['Consistent Good Sleepers','Short but Regular','Chronic Short & Variable','Variable Long Sleepers']
    zn = zn.reindex([r for r in ORDER if r in zn.index])

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor='white')
    ax.set_facecolor('white')
    im = ax.imshow(zn.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(fl))); ax.set_xticklabels(fl, fontsize=10)
    ax.set_yticks(range(len(zn))); ax.set_yticklabels(zn.index, fontsize=10)
    ax.tick_params(left=False, bottom=False)
    for i in range(len(zn)):
        for j, feat in enumerate(fc):
            raw = z.loc[zn.index[i], feat]; zv = zn.values[i,j]
            fmt = f'{raw:.0%}' if feat=='pct_short' else (f'{raw:,.0f}' if feat=='mean_steps' else f'{raw:.1f}')
            ax.text(j, i, fmt, ha='center', va='center', fontsize=9,
                    color='white' if abs(zv)>1.2 else DGRAY, fontweight='bold')
    plt.colorbar(im, ax=ax, label='z-score', fraction=0.02, pad=0.02)
    ax.set_title('Cluster Profiles (red = above average, blue = below average)', fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<div class="warning-box">⚠️ <b>Short but Regular</b>: 55% of nights under 6 hrs — severe chronic deprivation with a consistent schedule. High steps suggest trading sleep for activity.</div>', unsafe_allow_html=True)
    st.markdown('<div class="finding-box">💡 <b>Variable Long Sleepers</b>: Highest sleep but lowest steps and extreme variability. Likely shift workers, retirees, or those with irregular schedules.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — FAIRNESS
# ═══════════════════════════════════════════════════════════════════════════════
elif "Fairness" in page:
    st.title("Model Fairness")
    st.markdown("Does the model perform equally across demographic groups?")
    st.divider()

    target = st.radio("Target:", ["Sleep Duration","Sleep Consistency"], horizontal=True)
    tkey = "mean_sleep_hrs" if "Duration" in target else "iqr_sleep_hrs"
    sub = fair[fair.target==tkey].copy()
    overall = sub['R2'].mean()
    sub['delta'] = sub['R2'] - overall
    sub = sub.sort_values('R2', ascending=False)

    col1, col2 = st.columns([3,2])
    with col1:
        fig, ax = plt.subplots(figsize=(9,6), facecolor='white')
        ax.set_facecolor('white')
        colors = [RED if d < -0.005 else GREEN if d > 0.005 else GRAY for d in sub['delta']]
        bars = ax.barh(sub['subgroup'], sub['R2'], color=colors, alpha=0.85, edgecolor='white')
        ax.axvline(overall, color=DGRAY, lw=2, ls='--', alpha=0.7)
        ax.text(overall+0.001, len(sub)-0.5, f'avg {overall:.3f}', fontsize=9, color=DGRAY)
        for bar, val in zip(bars, sub['R2']):
            ax.text(val+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=9.5, color=DGRAY)
        ax.set_xlabel('R²', fontsize=11)
        ax.set_title(f'R² by Subgroup — {target}', fontsize=12, fontweight='bold', pad=10)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        ax.legend(handles=[mpatches.Patch(color=RED,label='Below avg'),
                            mpatches.Patch(color=GREEN,label='Above avg')],
                  fontsize=9, loc='lower right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("### Subgroup Results")
        disp = sub[['subgroup','N','R2']].copy()
        disp['R²'] = disp['R2'].round(3)
        disp['vs avg'] = disp['R2'].apply(lambda x: f"+{x-overall:.3f}" if x>overall else f"{x-overall:.3f}")
        st.dataframe(disp[['subgroup','N','R²','vs avg']], hide_index=True, use_container_width=True)

        ubr  = sub[sub.subgroup=='UBR']['R2'].values[0]
        wht  = sub[sub.subgroup=='White']['R2'].values[0]
        gap  = (wht - ubr) / wht * 100
        st.markdown(f'<div class="warning-box">⚠️ <b>{gap:.0f}% fairness gap</b><br>UBR R²={ubr:.3f} vs White R²={wht:.3f}.<br>The model explains substantially less for underrepresented minority participants.</div>', unsafe_allow_html=True)

        young = sub[sub.subgroup=='Age 18-40']['R2'].values[0]
        st.markdown(f'<div class="finding-box">📌 <b>18–40 year-olds</b> have the highest R²={young:.3f} — sleep is most behaviorally driven in younger adults.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Importance" in page:
    st.title("Feature Importance")
    st.markdown("Random Forest feature importances (mean decrease in impurity) trained on full cohort.")
    st.divider()

    target = st.radio("Target:", ["Sleep Duration","Sleep Consistency"], horizontal=True)
    imp = clean_imp(imp_d if "Duration" in target else imp_c)
    imp = imp[imp > 0.005]
    tgt_label = "Sleep Duration" if "Duration" in target else "Sleep Consistency (IQR)"

    col1, col2 = st.columns([3,2])
    with col1:
        simp = imp.sort_values()
        colors = [RED if i==simp.index[-1] else AMBER if i==simp.index[-2] else BLUE for i in simp.index]
        fig, ax = plt.subplots(figsize=(9, max(5,len(simp)*0.5)), facecolor='white')
        ax.set_facecolor('white')
        bars = ax.barh(simp.index, simp.values, color=colors, alpha=0.85, edgecolor='white', height=0.6)
        for bar, val in zip(bars, simp.values):
            ax.text(val+0.003, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=9, color=DGRAY)
        ax.set_xlabel('Feature Importance (RF)', fontsize=11)
        ax.set_title(f'Feature Importances — {tgt_label}', fontsize=12, fontweight='bold', pad=10)
        ax.xaxis.grid(True, alpha=0.3); ax.set_axisbelow(True); ax.tick_params(left=False)
        for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
        ax.set_xlim(0, simp.max()*1.25)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        st.markdown("### Top 5 Features")
        t5 = imp.head(5).reset_index(); t5.columns = ['Feature','Importance']
        t5['Importance'] = t5['Importance'].round(3)
        st.dataframe(t5, hide_index=True, use_container_width=True)
        st.divider()
        if "Duration" in target:
            st.markdown('<div class="finding-box">🔑 <b>BMI</b> and <b>female gender</b> dominate — biological factors drive sleep duration more than behavioral ones.</div>', unsafe_allow_html=True)
            st.markdown('<div class="finding-box">📌 <b>BMI × steps</b>: the effect of activity on sleep varies by body composition.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="finding-box">🔑 <b>Age × steps</b> (0.44): physical activity\'s effect on sleep regularity is strongly age-dependent — the single dominant predictor.</div>', unsafe_allow_html=True)
            st.markdown('<div class="finding-box">📌 <b>Step variability</b> (0.109): inconsistent activity → inconsistent sleep.</div>', unsafe_allow_html=True)
            st.markdown('<div class="finding-box">📌 <b>Self-rated health</b> (0.097): perceived wellbeing ties more to sleep regularity than duration.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Full Table")
    ft = imp.reset_index(); ft.columns=['Feature','Importance']; ft['Importance']=ft['Importance'].round(4)
    st.dataframe(ft, hide_index=True, use_container_width=True)
