import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# === 1) 读入数据 ===
df = pd.read_csv("/Users/cjy/Documents/code/dataset.csv")

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

# === 4) 定义 LASSO 选择函数 ===
def run_lasso(y_col, out_file):
    df_model = df.dropna(subset=[y_col] + X_cols).copy()
    X = df_model[X_cols]
    y = df_model[y_col]

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LASSO with cross-validation
    lasso = LassoCV(cv=5, random_state=42).fit(X_scaled, y)

    # 系数
    coef = pd.Series(lasso.coef_, index=X_cols)
    selected = coef[coef != 0]

    # 保存结果
    with open(out_file, "w") as f:
        f.write(f"=== LASSO Results for {y_col} ===\n")
        f.write(f"Optimal alpha (lambda): {lasso.alpha_:.4f}\n\n")
        f.write("Selected variables (non-zero coefficients):\n")
        f.write(str(selected))
        f.write("\n\nAll coefficients:\n")
        f.write(str(coef))

    print(f"✅ LASSO {y_col} 完成，结果已保存: {out_file}")
    return lasso, selected

# === 5) 分别跑 Europe & Americas ===
lasso_eu, sel_eu = run_lasso("Europe_Revenue",
    "/Users/cjy/Documents/code/lasso_results_europe.txt")

lasso_us, sel_us = run_lasso("Americas_Revenue",
    "/Users/cjy/Documents/code/lasso_results_americas.txt")
