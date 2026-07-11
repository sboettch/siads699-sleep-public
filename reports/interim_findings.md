# SIADS 699 — Interim Findings Report
**Team Sleep Deprived | July 2026**

---

## 1. Cohort Description

| Metric | Value |
|--------|-------|
| Total participants | 59,757 |
| Data source | All of Us CDR v9 (cdrv9, C2025Q4R6) |
| Sleep data | Fitbit `sleep_daily_summary` (main sleep only, ≥4 nights, 4–12 hrs) |
| Feature matrix | 25 variables across sleep, activity, demographics, BMI |

**Demographics:**
- Mean age: 56.9 years (SD 16.2); range 20–120
- Female: 66.3% | Male: 30.9% | Non-binary/other: 2.8%
- White: 70.4% | Black/African American: 7.5% | Asian: 4.8% | Multiracial: 6.9%
- UBR (underrepresented in biomedical research): 19.6%

**Data quality:**
- BMI missing: 9.1% (imputed with median in models)
- Steps missing: 0.7%

---

## 2. Key Sleep Statistics

| Metric | Value |
|--------|-------|
| Mean sleep duration | **6.79 hrs** (SD ~1.1) |
| Median sleep duration | ~6.8 hrs |
| Nights < 6 hrs (avg per participant) | **30%** |
| Sleep consistency (IQR) | ~1.3 hrs |
| Mean daily steps | 6,852 (SD 3,347) |

30% of nights averaged below 6 hours across the cohort — substantially below the recommended 7–9 hours. This is consistent with published All of Us sleep studies.

---

## 3. Model Performance (Phase 1)

Targets: **mean_sleep_hrs** (duration) and **iqr_sleep_hrs** (consistency).

| Model | Duration R² | Duration MAE | Consistency R² | Consistency MAE |
|-------|:-----------:|:------------:|:--------------:|:---------------:|
| Ridge | 0.071 | 0.546 hrs | 0.136 | 0.424 hrs |
| Lasso | 0.070 | 0.547 hrs | 0.127 | 0.427 hrs |
| Random Forest | **0.085** | **0.542 hrs** | **0.159** | **0.419 hrs** |

Low R² (7–16%) is expected — sleep is driven by unmeasured factors (genetics, stress, medications). Lifestyle factors explaining 7–16% of variance is consistent with epidemiology literature. Sleep consistency is more predictable than duration.

---

## 4. Fairness Findings

| Subgroup | N | Duration R² | Consistency R² |
|----------|---|:-----------:|:--------------:|
| Female | 35,524 | 0.108 | 0.216 |
| Male | 17,224 | 0.114 | 0.201 |
| UBR | 11,695 | **0.075** | 0.171 |
| White | 37,879 | 0.125 | 0.235 |
| Age 18–40 | 10,431 | **0.187** | 0.170 |
| Age 61–80 | 22,409 | 0.109 | 0.217 |

**Key finding:** UBR participants show a 40% relative gap in model accuracy vs White participants for sleep duration prediction. The model explains less variance for underrepresented groups — a limitation to report explicitly.

---

## 5. Limitations

1. Cross-sectional design — no causal claims
2. Wearable bias — Fitbit owners are self-selected
3. Fairness gap for UBR populations
4. Missing variables: stress, mental health, medications not yet included
5. Participant-level averages mask night-to-night dynamics

---

*Aggregate statistics only. No individual-level data exported from Workbench.*
