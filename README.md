

```markdown
# 👜 Chanel Predictive Variables

A reproducible Python pipeline modeling **Chanel’s regional revenues** using macroeconomic indicators from **FRED** and **Eurostat**.  
It compares **OLS (baseline)** and **LASSO (regularized)** regressions to identify the most influential predictors across the Americas and Europe.

---

## 📘 Overview

- Clean and merge macroeconomic + Chanel data  
- Run OLS baseline regression for each region  
- Run LASSO regression for feature selection  
- Store outputs as `.txt` logs for reproducibility  

---

## 🧭 Project Structure

```

.
├── baseline.py                      # Baseline OLS regression model
├── baseline_adjust.py               # Adjusted/sensitivity analysis version
├── clean.py                         # Data cleaning and dataset merging
├── lasso.py                         # LASSO feature selection & regression
├── test.py                          # Debugging/testing script
│
├── fred_macro_data_final.csv        # Final macro dataset (FRED / Eurostat)
├── fred_with_chanel_optionA.csv     # Merged macro + Chanel dataset (CSV)
├── fred_with_chanel_optionA.xlsx    # Merged macro + Chanel dataset (Excel)
│
├── baseline_regression_results.txt  # Baseline OLS output
├── baseline_adj.txt                 # Adjusted OLS output summary
├── lasso.txt                        # LASSO regression coefficients
└── README.md                        # Project documentation

````

---

## ⚙️ Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scipy scikit-learn statsmodels matplotlib
# optional for automated data pull
pip install fredapi python-dotenv
````

---

## ▶️ Run Scripts

```bash
python clean.py             # Prepare merged dataset
python baseline.py          # Baseline OLS regression
python baseline_adjust.py   # Adjusted OLS model
python lasso.py             # LASSO regression
```

---

## 📊 Baseline Regression Summary

**Americas Revenue (OLS)**

* R² = 0.872
* Significant predictors:

  * Trade_Weighted_USD_Index (−)
  * log_US_Real_GDP (−)
  * US_Consumer_Sentiment (−)
  * log_EU_Real_GDP (+)

**Europe Revenue (OLS)**

* R² = 0.957
* Significant predictors:

  * Trade_Weighted_USD_Index (−)
  * log_US_Real_GDP (−)
  * log_US_PCE (+)
  * US_CPI (+)
  * log_EU_Retail_Sales (−)
  * EU_Consumer_Confidence (−)

---

## 🧩 LASSO Regression Summary

**Americas Revenue**

```
Optimal alpha: 4.2068
Selected variables:
Trade_Weighted_USD_Index   -38.15
log_US_Real_GDP            -10.46
US_Consumer_Sentiment      -47.59
US_Federal_Funds_Rate       59.61
log_EU_Retail_Sales         73.78
EU_Consumer_Confidence     -40.57
```

**Europe Revenue**

```
Optimal alpha: 17.0271
Selected variables:
US_CPI                    +304.35
EU_ECB_Policy_Rate        +311.77
log_EU_Retail_Sales        +32.03
EU_Consumer_Confidence     -26.11
Trade_Weighted_USD_Index   -0.26
```

---

## 🧠 Interpretation

| Region   | Strong Negative Predictors                   | Strong Positive Predictors         |
| :------- | :------------------------------------------- | :--------------------------------- |
| Americas | USD Index, Consumer Sentiment, EU Confidence | Fed Funds Rate, EU Retail Sales    |
| Europe   | USD Index, EU Confidence                     | CPI, ECB Policy Rate, Retail Sales |

---

## 🔮 Future Work

* Add cross-validation and rolling forecasts
* Automate FRED API data collection
* Build interactive dashboards (Streamlit / Plotly)
* Extend model to Asia-Pacific markets

---

## 👩‍💻 Author

**Mi-Qin (Tina) Chen**
📧 [mc3208a@american.edu](mailto:mc3208a@american.edu)
🎓 American University — School of International Service
Focus: International Political Economy & Data-Driven Luxury Market Analytics
Supervised by **Prof. Krista Tuomi**

---

## 🪪 License

This project is for **academic and research use**.
Reuse and adaptation permitted with attribution.

```

---

🟢 **说明**：  
这份 README 是“纯代码可复制”版本，即整段粘贴到 `README.md` 后即可直接使用，不需要额外排版。  
是否希望我帮你生成一个“更简版（例如用于 GitHub 项目简介区域）”的 2–3 段摘要版？
```
