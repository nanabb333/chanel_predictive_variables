import pandas as pd
import numpy as np
import statsmodels.api as sm

# === 1) 读取数据 ===
df = pd.read_csv("/Users/tauras/Documents/Code/usbond/fred_with_chanel_optionA.csv")

# === 2) 对数化大数额变量 ===
for col in ["US_Real_GDP", "US_PCE", "EU_Real_GDP", "EU_Retail_Sales"]:
    df[f"log_{col}"] = np.log(df[col])

# === 3) baseline X 组合 ===
X_cols = [
    "Trade_Weighted_USD_Index",
    "log_US_Real_GDP",
    "log_US_PCE",
    "US_CPI",
    "US_Consumer_Sentiment",
    "US_Federal_Funds_Rate",
    "log_EU_Real_GDP",
    "log_EU_Retail_Sales",
    "EU_HICP",
    "EU_Consumer_Confidence",
    "EU_ECB_Policy_Rate"
]

# === 4) 建模函数 ===
def run_clustered_regression(y_col):
    df_model = df.dropna(subset=[y_col] + X_cols).copy()
    X = sm.add_constant(df_model[X_cols])
    y = df_model[y_col]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_model["Year"]})
    print(f"\n=== Baseline 回归结果: {y_col} ===")
    print(model.summary())
    return model

# === 5) 分别跑 Europe & Americas ===
model_europe   = run_clustered_regression("Europe_Revenue")
model_americas = run_clustered_regression("Americas_Revenue")

with open("/Users/tauras/Documents/Code/usbond/baseline_regression_results.txt", "w") as f:
    f.write("=== Europe Revenue Regression ===\n")
    f.write(str(model_europe.summary()))
    f.write("\n\n=== Americas Revenue Regression ===\n")
    f.write(str(model_americas.summary()))

print("✅ 回归结果已保存到 baseline_regression_results.txt")
