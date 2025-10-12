import pandas as pd
from fredapi import Fred
import os


try:
    fred_key = os.environ.get('FRED_API_KEY', '70d8df993b0bcf36557f9f678d98277c')
    fred = Fred(api_key=fred_key)
except Exception as e:
    exit()



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


    'EU_GDP': 'CPMNACSCAB1GQEA19',             
    'EU_Real_GDP': 'CLVMNACSCAB1GQEA19',       
    'EU_Retail_Sales': 'EA19SLRTTO02IXOBSAM',  
    'EU_HICP': 'CP0000EZ19M086NEST',
    'EU_Consumer_Confidence': 'CSESFT02EZM460S',
    'EU_ECB_Policy_Rate': 'ECBMRRFR',
    'EU_German_10Y_Bond_Yield': 'IRLTLT01DEM156N',

}


start_date = '2021-01-01'
end_date = '2023-10-01'


final_fred_df = pd.DataFrame()

for name, series_id in fred_series_to_fetch.items():
    try:
        data = fred.get_series(series_id, start_date=start_date, end_date=end_date)
        df_temp = pd.DataFrame(data, columns=[name])
        if final_fred_df.empty:
            final_fred_df = df_temp
        else:
            final_fred_df = final_fred_df.join(df_temp, how='outer')
    except Exception as e:
        print(Error)

final_fred_df = final_fred_df.ffill().bfill()

output_filename = '/Users/cjy/Documents/code/fred_macro_data_final.csv'
final_fred_df.to_csv(output_filename)
