# Chanel Predictive Variables

A lightweight Python project for modeling **Chanel’s financial performance** using macroeconomic indicators (mainly from FRED and Eurostat).  
The goal is to identify predictive relationships between global economic trends and Chanel’s key financial metrics.

---

## Overview

This repository demonstrates a reproducible workflow that:
1. **Cleans and merges macroeconomic datasets**
2. **Applies baseline OLS regression** for initial interpretability
3. **Tests LASSO regression** for variable selection and robustness
4. **Logs key outputs** for later comparison and visualization

---

## Project Structure


.
├── baseline.py # Baseline OLS regression model
├── baseline_adjust.py # Adjusted / sensitivity version
├── clean.py # Data cleaning and merging
├── lasso.py # LASSO feature selection & regression
├── test.py # Quick debugging & script testing
├── fred_macro_data_final.csv # Core macro dataset (FRED / Eurostat)
├── fred_with_chanel_optionA.csv # Merged macro + Chanel dataset (CSV)
├── fred_with_chanel_optionA.xlsx # Merged dataset in Excel format
├── baseline_regression_results.txt # Output log for baseline model
├── baseline_adj.txt # Output log for adjusted model
├── lasso.txt # Output log for LASSO model
└── README.md


---

##  Objective

This project explores **how macroeconomic indicators influence luxury brand performance** — specifically Chanel.  
By aligning publicly available macro data with Chanel’s financial data, the pipeline aims to:
- Quantify relationships between global consumption variables and brand performance
- Test statistical significance of selected predictors
- Compare OLS vs. LASSO in predictive strength and interpretability

---

## Data Sources

| Source | Description | Example Variables |
|:--------|:-------------|:-----------------|
| FRED (U.S.) | U.S. and international macroeconomic indicators | CPI, inflation rate, unemployment, consumer spending |
| Eurostat | EU household and business statistics | Household saving rate, consumption index |
| Chanel Reports | Aggregated financial data (2020–2024) | Sales revenue, operating margin, net profit |

All preprocessed datasets are already included for reproducibility.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/nanabb333/chanel_predictive_variables.git
cd chanel_predictive_variables

```

Future Work

Add cross-validation and time-series forecasting (ARIMA/XGBoost)
Automate FRED API pulls for real-time data updates
Integrate visual dashboards (Plotly / Streamlit) for interactive analysis
Compare with other brands for sector benchmarking

Author

Mi-Qin (Tina) Chen mc3208a@american.edu
American University · School of International Service
Focus: International Political Economy & Data-Driven Luxury Market Analytics

This project is under supervision of Prof. Krista Tuomi 