import pandas as pd
from pathlib import Path

fred_path = Path("/Users/cjy/Documents/code/dataset.csv")
fred = pd.read_csv(fred_path)

fred["DATE"] = pd.to_datetime(fred["DATE"], dayfirst=True, errors="coerce")
fred = fred.dropna(subset=["DATE"])

fred["Year"] = fred["DATE"].dt.year

chanel = pd.DataFrame({
    "Year": [2021, 2022, 2023, 2024],
    "Europe_Revenue":   [4042, 4720, 5606, 5676],
    "Americas_Revenue": [3529, 3859, 3960, 3790]
})

chanel["Europe_Growth"]   = chanel["Europe_Revenue"].pct_change()
chanel["Americas_Growth"] = chanel["Americas_Revenue"].pct_change()

merged = fred.merge(chanel, on="Year", how="left")
merged = merged[merged["Year"].between(2021, 2024)]

out_csv = Path("/Users/cjy/Documents/code/dataset.csv")

merged.to_csv(out_csv, index=False)



