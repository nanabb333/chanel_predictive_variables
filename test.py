import pandas as pd
from fredapi import Fred
import os

# --- 1. 初始化 FRED API ---
try:
    # 建议将 API Key 设置为环境变量 FRED_API_KEY
    fred_key = os.environ.get('FRED_API_KEY', '70d8df993b0bcf36557f9f678d98277c')
    fred = Fred(api_key=fred_key)
    print("FRED API 初始化成功。")
except Exception as e:
    print(f"FRED API 初始化失败，请检查你的 API 密钥。错误: {e}")
    exit()


# --- 2. 定义 FRED 指标列表 (不含上证指数) ---
fred_series_to_fetch = {
    'USD_CNY_Rate': 'DEXCHUS',
    'Trade_Weighted_USD_Index': 'DTWEXBGS',
    'US_GDP': 'GDP',
    'US_Real_GDP': 'GDPC1',
    'US_PCE': 'PCE',
    'US_CPI': 'CPIAUCSL',
    'US_Personal_Saving_Rate': 'PSAVERT',
    'US_Consumer_Sentiment': 'UMCSENT',
    'US_SP500_Index': 'SP500',
    'US_Federal_Funds_Rate': 'DFF',
    'US_10Y_Treasury_Yield': 'DGS10',

        # --- 欧洲指标 ---
    'EU_GDP': 'CPMNACSCAB1GQEA19',             # 新增的欧洲指标
    'EU_Real_GDP': 'CLVMNACSCAB1GQEA19',       # ...
    'EU_Retail_Sales': 'EA19SLRTTO02IXOBSAM',  # 根据用户反馈更新
    'EU_HICP': 'CP0000EZ19M086NEST',
    'EU_Consumer_Confidence': 'CSESFT02EZM460S',
    'EU_ECB_Policy_Rate': 'ECBMRRFR',
    'EU_German_10Y_Bond_Yield': 'IRLTLT01DEM156N',

}

# --- 3. 设置时间范围 ---
start_date = '2021-01-01'
end_date = '2023-10-01'
print(f"\n设置 FRED 数据获取时间范围: {start_date} 到 {end_date}")

# --- 4. 获取数据 ---
print("\n开始从 FRED 获取数据...")
final_fred_df = pd.DataFrame()

for name, series_id in fred_series_to_fetch.items():
    try:
        data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
        df_temp = pd.DataFrame(data, columns=[name])
        if final_fred_df.empty:
            final_fred_df = df_temp
        else:
            final_fred_df = final_fred_df.join(df_temp, how='outer')
        print(f"成功获取: {name} ({series_id})")
    except Exception as e:
        print(f"获取失败: {name} ({series_id}). 错误: {e}")

# --- 5. 数据处理 ---
print("\n数据获取完成，进行初步处理...")
final_fred_df = final_fred_df.ffill().bfill()
print("数据处理完成。")

# --- 6. 保存数据 ---
output_filename = '/Users/cjy/Documents/code/fred_macro_data_final.csv'
final_fred_df.to_csv(output_filename)
print(f"\nFRED 数据已成功保存到文件: {output_filename}")