# Sleep Health & Lifestyle Factors
### SIADS 699 Capstone — Team Sleep Deprived

An analysis of the relationship between sleep behavior and lifestyle/environmental factors using Fitbit wearable data from the [NIH All of Us Research Program](https://www.researchallofus.org/).

**Team:** Sophia Boettcher (Lead Developer) · Auston Balwinski (Project Manager) · Jared Fox (Lead Visualizer) · Hunter Belous (Lead Analyst)

---

## Research Questions

- Which lifestyle and environmental factors are most strongly associated with sleep duration and consistency?
- How do physical activity, BMI, demographics, and socioeconomic conditions relate to sleep outcomes?

## Data

This project uses the [All of Us wearables dataset](https://doi.org/10.1038/s41591-026-04352-3) (Patten et al., 2026), which contains Fitbit-derived measurements from >59,000 participants. **All data access is through the All of Us Researcher Workbench (Controlled Tier). No individual-level data is stored in this repository.**

## Structure

```
├── notebooks/          # Analysis notebooks (designed to run in AoU Workbench)
├── src/                # Reusable Python modules
├── dashboard/          # Streamlit dashboard app
├── synthetic_data/     # Synthetic data matching AoU schema for local development
├── outputs/            # Aggregate results, figures (no participant-level data)
└── requirements.txt
```

## Reproducing the Analysis

All notebooks are designed to run inside the **All of Us Researcher Workbench**. To run locally for development/testing, use the synthetic dataset:

```bash
pip install -r requirements.txt
python synthetic_data/generate.py
jupyter notebook notebooks/01_data_pull.ipynb
```

## Ethics & Privacy

- All data governed by the All of Us Data Use and Registration Agreement
- No individual-level data appears in this repository
- All results reported as population-level aggregates
- Models evaluated for fairness across demographic subgroups

## Citation

Patten, T., Preble, E.A., Master, H. et al. The All of Us Research Program's wearables dataset. *Nat Med* 32, 2302–2310 (2026). https://doi.org/10.1038/s41591-026-04352-3
