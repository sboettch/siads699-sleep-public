# 😴 Sleep Health Prediction & Phenotyping in the *All of Us* Research Program

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![All of Us CDR v9](https://img.shields.io/badge/All%20of%20Us-CDR%20v9-purple?logo=nih)
![Fitbit Wearables](https://img.shields.io/badge/Fitbit-Wearables-teal)

> **SIADS 699 Capstone Project** · University of Michigan School of Information · July 2026  
> *Sophia Boettcher, Auston Balwinski, Hunter Belous, Jared Fox*

---

## 📖 Overview

This project applies machine learning to Fitbit-derived sleep data from **59,757 participants** in the NIH *All of Us* Research Program (CDR v9) to (1) predict individual sleep duration and consistency using sociodemographic, health, and physical activity features, (2) identify distinct sleep behavioral phenotypes via unsupervised clustering, and (3) evaluate predictive model fairness across racial/ethnic subgroups. The work reveals a **40% fairness gap** in model performance between White and Underrepresented in Biomedical Research (UBR) participants — a critical equity finding for algorithm-assisted sleep health interventions.

---

## 🔑 Key Findings

- **Best model:** HistGBM achieved R² = 0.099 (sleep duration) and R² = 0.184 (sleep consistency), a **+16% relative improvement** over Phase 1 linear baselines
- **Top predictor for duration:** Nights tracked (0.181), BMI (0.162), female gender (0.150)
- **Top predictor for consistency:** Age × steps interaction (0.442) — a novel engineered feature — dominates the consistency model, suggesting physical activity is especially stabilizing for older adults
- **4 sleep phenotypes identified:**
  - 🟢 **Consistent Good Sleepers** (42%, n=25,176) — adequate duration, low variability, high activity
  - 🔴 **Chronic Short & Variable** (24%, n=14,479) — highest-risk group; short + irregular sleep, lowest steps
  - 🟡 **Short but Regular** (24%, n=14,467) — constrained but predictable schedules
  - 🟣 **Variable Long Sleepers** (9%, n=5,635) — potential overlap with depression/chronic illness
- **Fairness gap:** UBR participants R² = 0.075 vs. White R² = 0.125 for duration — a **40% relative gap**
- **Cohort:** 59,757 participants; mean age 56.9, 66.3% female, 70.4% White, 19.6% UBR; mean sleep 6.79 hrs/night

---

## 📁 Repository Structure

```
siads699-sleep-public/
├── README.md
├── dashboard/
│   ├── app.py                  # Streamlit dashboard
│   ├── requirements.txt
│   └── *.csv                   # Aggregate result files
├── reports/
│   ├── interim_findings.md
│   ├── final_report.md
│   └── figures/               # All publication figures
└── notebooks/                 # Coming soon: cleaned public notebooks
```

---

## 🚀 Running the Dashboard

The interactive Streamlit dashboard visualizes model results, cluster profiles, and fairness metrics.

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`. No authentication required — all data displayed is aggregate only.

---

## 🔬 Methods Overview

| Component | Details |
|---|---|
| **Data source** | *All of Us* CDR v9, `fitbit_sleep_daily_summary` |
| **Cohort** | 59,757 adults with ≥30 Fitbit nights + Basics survey |
| **Outcomes** | Mean sleep duration (hrs); IQR of nightly sleep (consistency) |
| **Phase 1 models** | Ridge, Lasso, ElasticNet (5-fold CV) |
| **Phase 2 models** | Random Forest, HistGBM (5-fold CV + RandomizedSearchCV) |
| **Feature engineering** | Interaction terms: age×steps, BMI×steps, female×age |
| **Clustering** | KMeans k=4; elbow + silhouette selection |
| **Fairness evaluation** | Stratified R² by race/ethnicity (White vs. UBR) |
| **Environment** | *All of Us* Researcher Workbench (Terra), Python 3.11 |

---

## 🗃️ Data Notice

> **⚠️ No individual-level data are included in this repository.**

All analyses were conducted within the secure [*All of Us* Researcher Workbench](https://workbench.researchallofus.org/) (Terra cloud environment). Access to the Controlled Tier dataset requires registration and approval through the *All of Us* Research Program.

This public repository contains **aggregate results only** (summary statistics, model coefficients, cluster centroids, and figures). CSV files in `dashboard/` contain only population-level summaries with no records that could identify individual participants.

---

## 📊 Publication Figures

All figures are located in `reports/figures/`. Key figures include:

| Filename | Description |
|---|---|
| `fig_fte_histogram.png` | Distribution of mean sleep duration |
| `fig_fte_dotplot_age.png` | Age distribution by sleep cluster |
| `fig_fte_importance_mean_sleep_hrs.png` | Feature importances — duration model |
| `fig_fte_importance_iqr_sleep_hrs.png` | Feature importances — consistency model |
| `fig_fte_cluster_heatmap.png` | Cluster profile heatmap |
| `fig_fte_slope_fairness.png` | Fairness gap: R² by race/ethnicity |

---

## 📝 Citation

If you use or build on this work, please cite:

```bibtex
@misc{boettcher2026sleep,
  author       = {Boettcher, Sophia and Balwinski, Auston and Belous, Hunter and Fox, Jared},
  title        = {Predicting and Phenotyping Sleep Health in a Diverse National Cohort:
                  Evidence from the All of Us Research Program},
  year         = {2026},
  howpublished = {SIADS 699 Capstone, University of Michigan},
  url          = {https://github.com/sophiaboettcher/siads699-sleep-public}
}
```

---

## 🙏 Acknowledgments

This research was made possible by the *All of Us* Research Program participants, whose generous data contributions power health equity science. The *All of Us* Research Program is supported by the National Institutes of Health, Office of the Director: Regional Medical Centers: 1 OT2 OD026549; 1 OT2 OD026554; 1 OT2 OD026557; 1 OT2 OD026556; 1 OT2 OD026550; 1 OT2 OD 026552; 1 OT2 OD026553; 1 OT2 OD026548; 1 OT2 OD026551; 1 OT2 OD026555; IAA #: AOD 16037; Federally Qualified Health Centers: HHSN 263201600085U; Data and Research Center: 5 U2C OD023196; Biobank: 1 U24 OD023121; The Participant Center: U24 OD023176; Participant Technology Systems Center: 1 U24 OD023163; Communications and Engagement: 3 OT2 OD023205; 3 OT2 OD023206; and Community Partners: 1 OT2 OD025277; 3 OT2 OD025315; 1 OT2 OD025337; 1 OT2 OD025276.

Additional thanks to the University of Michigan School of Information SIADS 699 teaching team for their guidance throughout this capstone.

---

*Licensed under the [MIT License](LICENSE). See LICENSE for details.*
