import pandas as pd 
import numpy as np
import config   


raw_store_sales = pd.read_csv(config.RAW_SALES_PATH)
raw_store_features = pd.read_csv(config.RAW_FEATURES_PATH)
raw_stores = pd.read_csv(config.RAW_STORES_PATH)


# the date is in day/month/year format i.e d/m/Y
raw_store_sales['Date'] = pd.to_datetime(raw_store_sales['Date'], format='%d/%m/%Y')
raw_store_sales = raw_store_sales.sort_values(by=['Date', 'Store'], ascending=True).reset_index(drop=True)

# Current sales data
past_data = raw_store_sales[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()

# Add 52 weeks to the current dates , everything else remains the same 
past_data['Date'] = past_data['Date'] + pd.Timedelta(weeks=52)
past_data = past_data.rename(columns={'Weekly_Sales':'Baseline_Prediction'})

baseline_predictions = raw_store_sales.merge(past_data, on=['Store','Dept','Date'], how='left')
# drop the 2010 dates , because we dont have 2009 data for them to generate predictions
baseline_predictions = baseline_predictions.dropna(subset=['Baseline_Prediction'])

# RAW Absolute error
baseline_predictions['ABS_Error'] = abs(baseline_predictions['Weekly_Sales'] - baseline_predictions['Baseline_Prediction'])
baseline_predictions['Weight'] = np.where(baseline_predictions['IsHoliday'], 5, 1)

# weighted mean absolute error
wmae = ((baseline_predictions['Weight'] *
        baseline_predictions['ABS_Error']).sum() /
        baseline_predictions['Weight'].sum())

# weighted mean absolute percentage error
wmape = (((baseline_predictions['Weight'] * 
        baseline_predictions['ABS_Error']).sum()) /
        (baseline_predictions['Weight'] * 
        baseline_predictions['Weekly_Sales']).sum())


print(f"WMAE: {wmae:.2f}")
print(f"WMAPE: {(wmape * 100):.2f}%")