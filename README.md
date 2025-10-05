👜 Macroeconomic Drivers of Chanel's Revenue
This repository contains a data-driven econometric model that links Chanel’s regional financial performance to key macroeconomic indicators from the United States (FRED) and the Euro Area.

It implements and compares Ordinary Least Squares (OLS) and LASSO (L1 Regularization) regression models to identify the most influential predictors of Chanel’s sales in its Americas and Europe regions between 2021 and 2024.

📘 Methodology
The analytical process is structured as follows:

Data Collection: Macroeconomic data (e.g., GDP, CPI, interest rates, consumer sentiment) is fetched from the Federal Reserve Economic Data (FRED) database using their official API.

Data Preprocessing: The raw FRED data is merged with Chanel's annual regional revenue data. Key variables with large magnitudes (like GDP and PCE) are log-transformed to stabilize variance.

Baseline Modeling: An initial OLS regression is performed using a comprehensive set of macroeconomic variables to establish a baseline model. Standard errors are clustered by year to account for time-based correlations.

Feature Selection: LASSO regression is employed to automatically select the most influential variables, shrinking the coefficients of less important factors to zero.

Refined Modeling: A final OLS model is built using only the statistically significant variables identified from the baseline model or the variables selected by LASSO to create a more parsimonious and interpretable model.

🧭 Project Structure
.
├── data/
│   └── fred_with_chanel_optionA.csv      # Merged dataset with FRED and Chanel data
│
├── results/
│   ├── baseline_regression_results.txt   # Output from the full baseline OLS model
│   ├── baseline_adj.txt                  # Output from the refined OLS model
│   └── lasso.txt                         # Variables selected by LASSO regression
│
├── test.py                           # Fetches raw data from the FRED API
├── clean.py                          # Cleans and merges macroeconomic + Chanel data
├── baseline.py                       # Runs baseline OLS regressions for Americas and Europe
├── baseline_adjust.py                # Runs OLS with only statistically significant variables
└── lasso.py                          # Runs LASSO regression for variable selection


🚀 How to Run the Analysis
To replicate the results, follow these steps:

1. Prerequisites:

Python 3.x

Install the required libraries:

pip install pandas numpy statsmodels scikit-learn fredapi

2. FRED API Key:

Obtain a free API key from the FRED website.

Set it as an environment variable.

On macOS/Linux: export FRED_API_KEY='your_key_here'

On Windows: set FRED_API_KEY='your_key_here'

Alternatively, you can hardcode the key directly in the test.py script.

3. Run the Scripts in Order:
The scripts should be executed in the following sequence. The result files will be generated in the results/ directory.

# 1. Fetch the latest data from FRED
python test.py

# 2. Clean and merge the data
python clean.py

# 3. Run the regression models
python baseline.py
python baseline_adjust.py
python lasso.py

🧠 Interpretation Summary
Region

Strong Negative Predictors

Strong Positive Predictors

Americas

USD Index, US Consumer Sentiment, EU Confidence

US Federal Funds Rate, EU Retail Sales

Europe

USD Index

US CPI

The results indicate that Chanel's regional performance is highly sensitive to a mix of currency strength (USD Index), consumer confidence, and monetary policy (interest rates and inflation). The models reveal a notable interdependence between U.S. economic conditions and European consumer dynamics in shaping luxury sales.

🔍 Future Extensions
Incorporate cross-validation and rolling forecast methodologies for more robust model validation.

Extend the model to cover Chanel's Asia-Pacific revenue as more granular data becomes available.

Develop an interactive dashboard using Streamlit or Plotly to visualize the results and run new scenarios.

👩‍💻 Author
Mi-Qin (Tina) Chen

📧 mc3208a@american.edu

🎓 American University — School of International Service

Focus: International Political Economy & Data-Driven Luxury Market Analytics

Supervised by Prof. Krista Tuomi.

🪪 License
This project is intended for academic and research purposes. You may reuse or modify the code and analysis with proper attribution.

