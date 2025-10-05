

```markdown
# 👜 Chanel Predictive Variables

This repository builds a **data-driven econometric model** linking Chanel’s regional financial performance to macroeconomic indicators from the **U.S. (FRED)** and **Euro Area (Eurostat)**.  
It implements and compares **OLS (baseline)** and **LASSO (regularized)** regressions to identify the most influential predictors of Chanel’s sales in the Americas and Europe.

---

## 📘 Overview

The project consists of three main analytical layers:

1. **Data Preparation** — Cleaning and merging macroeconomic and Chanel datasets  
2. **Baseline OLS Regression** — Estimating the relationships between macro variables and Chanel revenues  
3. **LASSO Regression** — Selecting the most impactful predictors through regularization and cross-validation  

The results are fully reproducible and stored as plain-text outputs (`.txt`), enabling quick review and replication.

---

## 🧭 Project Structure

```

.
├── baseline.py                      # Runs baseline OLS regressions for Americas and Europe
├── baseline_adjust.py               # Alternative baseline with refined variables & diagnostics
├── lasso.py                         # LASSO regression for variable selection (sklearn)
├── clean.py                         # Cleans and merges macroeconomic + Chanel data
├── test.py                          # Scratchpad / debugging script
│
├── fred_macro_data_final.csv        # Final macro dataset (FRED + Eurostat)
├── fred_with_chanel_optionA.csv     # Combined Chanel + macro dataset (CSV)
├── fred_with_chanel_optionA.xlsx    # Same dataset in Excel format
│
├── baseline_regression_results.txt  # Baseline regression output
├── baseline_adj.txt                 # Adjusted OLS output summary
├── lasso.txt                        # LASSO regression coefficients & alpha values
└── README.md                        # Project documentation

````

---

## 🎯 Objective

This project investigates **how global macroeconomic variables explain Chanel’s regional revenue performance**.  
By modeling **Americas** and **Europe** revenue series separately, it seeks to uncover:

- The macro indicators most correlated with Chanel’s performance  
- The direction and magnitude of those relationships  
- Whether predictive accuracy improves using **regularized (LASSO)** regression  

---

## 📊 Key Findings

### Baseline OLS Models:contentReference[oaicite:0]{index=0}

**Americas Revenue**
- \( R^2 = 0.872 \): High explanatory power  
- Significant predictors:  
  - **Trade_Weighted_USD_Index (−)** → Dollar strength negatively affects sales  
  - **log_US_Real_GDP (−)** → U.S. real output growth inversely related, likely reflecting post-pandemic base effects  
  - **US_Consumer_Sentiment (−)** → Weaker confidence reduces luxury spending  
  - **log_EU_Real_GDP (+)** → Cross-region demand correlation  

**Europe Revenue**
- \( R^2 = 0.957 \): Very strong model fit  
- Significant predictors:  
  - **Trade_Weighted_USD_Index (−)**  
  - **US_CPI (+)**  
  - **log_US_Real_GDP (−)**  
  - **log_US_PCE (+)**  
  - **log_EU_Retail_Sales (−)**  
  - **EU_Consumer_Confidence (−)**  

These results suggest that both **transatlantic macro trends** and **domestic consumer sentiment** substantially shape Chanel’s performance across regions.

---

### LASSO Regression Results:contentReference[oaicite:1]{index=1}

**Americas Revenue**
- Optimal λ = 4.2068  
- Key predictors retained:  
  - `Trade_Weighted_USD_Index` (−38.15)  
  - `US_Federal_Funds_Rate` (+59.61)  
  - `log_EU_Retail_Sales` (+73.78)  
  - `US_Consumer_Sentiment` (−47.59)  
  - `EU_Consumer_Confidence` (−40.57)

**Europe Revenue**
- Optimal λ = 17.0271  
- Key predictors retained:  
  - `US_CPI` (+304.35)  
  - `EU_ECB_Policy_Rate` (+311.77)  
  - `EU_Consumer_Confidence` (−26.11)  
  - `log_EU_Retail_Sales` (+32.03)

📈 **Interpretation:**  
Luxury revenue is sensitive to **monetary conditions**, **inflation expectations**, and **consumer confidence** across both continents.  
The LASSO model refines variable selection, removing noise while keeping cross-regional linkages evident.

---

## ⚙️ Setup & Usage

### 1️⃣ Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scipy scikit-learn statsmodels matplotlib
````

(Optional for FRED API automation:)

```bash
pip install fredapi python-dotenv
```

### 2️⃣ Run Scripts

```bash
python clean.py             # Prepare merged dataset
python baseline.py          # Baseline OLS regression
python baseline_adjust.py   # Adjusted OLS model
python lasso.py             # LASSO variable selection
```

Each script generates its own `.txt` output log in the project root.

---

## 🧠 Interpretation Summary

| Region       | Strong Negative Predictors                   | Strong Positive Predictors         |
| :----------- | :------------------------------------------- | :--------------------------------- |
| **Americas** | USD Index, Consumer Sentiment, EU Confidence | Fed Funds Rate, EU Retail Sales    |
| **Europe**   | USD Index, EU Confidence                     | CPI, ECB Policy Rate, Retail Sales |

Both regions reveal the **interdependence between U.S. monetary policy and European consumer dynamics** in shaping Chanel’s luxury sales performance.

---

## 🔍 Future Extensions

* Add **cross-validation** and **rolling forecast** performance metrics
* Automate **real-time FRED pulls** via API
* Build **interactive dashboards** (Streamlit / Plotly)
* Extend coverage to **Asia-Pacific revenue** once data availability improves

---

## 👩‍💻 Author

**Mi-Qin (Tina) Chen**
📧 [mc3208a@american.edu](mailto:mc3208a@american.edu)
🎓 *American University — School of International Service*
**Focus:** International Political Economy & Data-Driven Luxury Market Analytics

Supervised by **Prof. Krista Tuomi**

---

## 🪪 License

This project is intended for **academic and research purposes**.
You may reuse or modify it with proper attribution.

```

