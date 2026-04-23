TARGET = 'weekly_sales' 
HOLIDAY_COL = 'isholiday'

NUMERIC_FEATURES = [
    'store_size', 'temperature', 'fuel_price', 'cpi', 'unemployment', 
    'markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5', 'total_markdown',
    'lag_1_log', 'lag_5_log', 'lag_52_log', 'sales_last_year', 'rolling_4_wk_log_sales_avg', 'cpi_lag_3_month'
]

# Logic: IDs and categorical states that need OHE for Linear, and Native Splitting for Trees
CATEGORICAL_FEATURES = [
    'store', 'dept', 'store_type', 'isholiday', 'month', 'week_of_year'
]

# Logic: Already perfect numerical continuous boundaries (-1 to 1)
PASSTHROUGH_FEATURES = [
    'sin_week', 'cos_week'
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PASSTHROUGH_FEATURES
TRAIN_SPLIT_QUANTILE = 0.70
VAL_SPLIT_QUANTILE = 0.85

XGBOOST_PARAMS = {
    'n_estimators': 500

}