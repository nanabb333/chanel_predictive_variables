# merge_chanel_with_fred_optionA.py
import pandas as pd
from pathlib import Path

# === 1) 读取 FRED 宏观数据 ===
fred_path = Path("/Users/cjy/Documents/code/dataset.csv")
fred = pd.read_csv(fred_path)

# 注意 DATE 格式是 dd/mm/yyyy，需要 dayfirst=True
fred["DATE"] = pd.to_datetime(fred["DATE"], dayfirst=True, errors="coerce")
fred = fred.dropna(subset=["DATE"])

# 提取年份
fred["Year"] = fred["DATE"].dt.year

# === 2) Chanel 地区收入数据（年度）===
chanel = pd.DataFrame({
    "Year": [2021, 2022, 2023, 2024],
    "Europe_Revenue":   [4042, 4720, 5606, 5676],
    "Americas_Revenue": [3529, 3859, 3960, 3790]
})

# 增速（同比 %）
chanel["Europe_Growth"]   = chanel["Europe_Revenue"].pct_change()
chanel["Americas_Growth"] = chanel["Americas_Revenue"].pct_change()

# === 3) 合并：不做年度聚合，直接把年度 Y merge 到每个月/季度 X ===
merged = fred.merge(chanel, on="Year", how="left")
merged = merged[merged["Year"].between(2021, 2024)]

# === 4) 保存结果（CSV + Excel 两份）===
out_csv = Path("/Users/cjy/Documents/code/dataset.csv")

merged.to_csv(out_csv, index=False)
merged.to_excel(out_xlsx, index=False)

print("✅ 合并完成")
print("CSV 文件:", out_csv)
print("Excel 文件:", out_xlsx)
print(merged.head())
