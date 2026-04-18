import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
import config
from structlog import getLogger

logger = getLogger()

from typing import Any

# 1. Custom Evaluation Metric
def wmape_metric(y_true: Any, y_pred: Any, is_holiday: Any) -> float:
    """
    Calculate the Weighted Mean Absolute Percentage Error (WMAPE).
    
    Holiday weeks are given 5x weight compared to normal weeks to reflect
    the higher business impact of stockouts during promotional periods.

    Args:
        y_true (np.ndarray): Array of actual sales values.
        y_pred (np.ndarray): Array of predicted sales values.
        is_holiday (np.ndarray): Binary array indicating if a week is a holiday (1) or not (0).

    Returns:
        float: The calculated WMAPE score.
    """
    logger.debug("Calculating WMAPE metric", samples=len(y_true))
    # Holiday weeks are weighted 5x
    weights = np.where(is_holiday == 1, 5, 1)
    # WMAPE Formula: Sum of Absolute Errors / Sum of Actuals
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * y_true))

def run_feasibility_study() -> None:
    """
    Run a minimum viable ML pipeline to establish a predictive baseline.
    
    This function reads gold-layer data, handles missing values, splits it 
    chronologically, trains a naive Random Forest regressor, and evaluates it 
    using the WMAPE metric against a manual heuristic baseline.

    Raises:
        Exception: Propagates any error encountered during pipeline execution.
    """
    try:
        start_time = time.time()
        logger.info("Initializing ML Feasibility Study...", model="RandomForest")
        
        # 2. Load Data
        logger.info("Reading Gold Data", source="gold")
        df = pd.read_parquet(config.GOLD_MASTER_PATH)
        
        # 3. Handle NULLs (Time-Series MLOps best practice)
        # We mathematically drop the first 52 weeks (the warm-up period) rather than impute fake data.
        initial_len = len(df)
        df = df.dropna()
        logger.info("Dropped early 'warm-up' time-series rows to handle NULL lags", dropped=initial_len - len(df))
        
        # 4. Feature Selection
        features = [
            'store', 'dept', 'isholiday', 'store_size', 'temperature', 
            'fuel_price', 'cpi', 'unemployment', 'total_markdown', 
            'month', 'week_of_year', 'rolling_4_wk_sales_avg', 'sales_last_year', 'cpi_lag_3_month'
        ]
        target = 'weekly_sales'
        
        # 5. Chronological Train/Test Split
        # Data Leakage Warning: DO NOT use random train_test_split. Time-series requires splitting by continuous date.
        df = df.sort_values('date')
        split_date = df['date'].quantile(0.8)  # 80/20 chronological split
        
        train = df[df['date'] < split_date]
        test = df[df['date'] >= split_date]
        
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        
        logger.info("Executing Chronological Split", train_rows=len(train), test_rows=len(test), split_date=str(split_date))
        
        # 6. Train Random Forest (Naive Baseline)
        logger.info("Training naive Random Forest... (this may take a minute)")
        model = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)
        
        # 7. Predict & Evaluate
        preds = model.predict(X_test)
        
        # Calculate Custom WMAPE 
        error_wmape = wmape_metric(y_test.values, preds, test['isholiday'].values)
        
        logger.info('Model WMAPE', score=round(error_wmape, 4))
        logger.info('Target to Beat', manual_heuristic=0.1185)
        
        if error_wmape < 0.1185:
            logger.info("FEASIBILITY PASSED! We beat the baseline!", wmape=round(float(error_wmape)*100, 2), status="success")
        else:
            logger.error("FEASIBILITY FAILED! We did not beat the baseline.", status="failed")

        duration = round(time.time() - start_time, 2)
        logger.info("ML Feasibility Study completed", duration_seconds=duration)

    except Exception as e:
        logger.error("ML Feasibility Study failed", error=str(e))
        raise

if __name__ == "__main__":
    run_feasibility_study()