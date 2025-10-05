import pandas as pd
import numpy as np
import statsmodels.api as sm

# === 1) 读取数据 ===
df = pd.read_csv("/Users/cjy/Documents/code/fred_with_chanel_optionA.csv")

# === 2) 对数化大额变量 ===
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

# === 4) 筛选显著变量并重新跑回归 ===
def run_refined_regression(y_col, out_file):
    df_model = df.dropna(subset=[y_col] + X_cols).copy()
    X = sm.add_constant(df_model[X_cols])
    y = df_model[y_col]
    
    # 先跑 baseline
    model_full = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_model["Year"]})
    
    # 根据 p<0.1 筛选显著变量
    pvals = model_full.pvalues
    sig_vars = [var for var in pvals.index if var != "const" and pvals[var] < 0.1]
    
    if not sig_vars:
        sig_vars = X_cols  # 防止没选上，回退到全部
    
    X_refined = sm.add_constant(df_model[sig_vars])
    model_refined = sm.OLS(y, X_refined).fit(cov_type="cluster", cov_kwds={"groups": df_model["Year"]})
    
    # 保存结果
    with open(out_file, "w") as f:
        f.write(f"=== {y_col} Baseline 回归结果 ===\n")
        f.write(str(model_full.summary()))
        f.write("\n\n=== 精简版回归结果 (p<0.1) ===\n")
        f.write("保留的变量: " + ", ".join(sig_vars) + "\n\n")
        f.write(str(model_refined.summary()))
    
    print(f"✅ {y_col} 结果已保存: {out_file}")
    return model_full, model_refined, sig_vars

# === 5) 分别跑 Europe & Americas ===
model_eu_full, model_eu_refined, sig_eu = run_refined_regression(
    "Europe_Revenue",
    "/Users/cjy/Documents/code/regression_results_europe.txt"
)

model_us_full, model_us_refined, sig_us = run_refined_regression(
    "Americas_Revenue",
    "/Users/cjy/Documents/code/regression_results_americas.txt"
)
