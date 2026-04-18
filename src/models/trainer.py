import polars as pl
import pandas as pd
import numpy as np 
import config 
import config_ml 
from models.pipeline_factory import get_model_pipeline
import structlog 
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline

logger = structlog.get_logger()

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
        float: The calculated WMAPE score in percentage.
    """
    logger.debug("Calculating WMAPE metric", samples=len(y_true))
    weights = np.where(is_holiday== 1, 5, 1)

    return float(np.round((np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * y_true)) * 100.0, 2))

def prepare_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any, pd.DataFrame, pd.Series, Any]:
    """
    Load data from the Gold layer and split chronologically into train, validation, and test sets.

    Returns:
        Tuple: Contains the following elements in order:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target.
            - X_val (pd.DataFrame): Validation features.
            - y_val (pd.Series): Validation target.
            - is_holiday_val (np.ndarray): Validation holiday boolean array.
            - X_test (pd.DataFrame): Test features.
            - y_test (pd.Series): Test target.
            - is_holiday_test (np.ndarray): Test holiday boolean array.
    """
    import polars.selectors as cs
    logger.info("Loading Gold layer to memory...")
    lf = pl.scan_parquet(config.GOLD_DATA_PATH)
    # CRITICAL: Polars differentiates between float NaNs and SQL Nulls!
    # drop_nulls() correctly drops the entire first 52-week "warm up" period. 
    lf = lf.drop_nulls().sort('date')

    # Fix: Cast Integers to Floats to prevent MLflow/BentoML Schema enforce crashes on NaN prediction
    lf = lf.with_columns(cs.integer().cast(pl.Float64))

    for col in config_ml.CATEGORICAL_FEATURES:
        lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
    
    split_date_1 = lf.select(pl.col('date').quantile(config_ml.TRAIN_SPLIT_QUANTILE)).collect().item()
    split_date_2 = lf.select(pl.col('date').quantile(config_ml.VAL_SPLIT_QUANTILE)).collect().item()
    
    train_df = lf.filter(pl.col('date') <= split_date_1).collect().to_pandas()
    val_df = lf.filter((pl.col('date') > split_date_1) & (pl.col('date') <= split_date_2)).collect().to_pandas()
    test_df = lf.filter(pl.col('date') > split_date_2).collect().to_pandas()

    X_train = train_df[config_ml.FEATURES]
    y_train = train_df[config_ml.TARGET]

    X_val = val_df[config_ml.FEATURES]
    y_val = val_df[config_ml.TARGET]
    is_holiday_val = val_df[config_ml.HOLIDAY_COL].values

    X_test = test_df[config_ml.FEATURES]
    y_test = test_df[config_ml.TARGET]
    is_holiday_test = test_df[config_ml.HOLIDAY_COL].values

    logger.info("Chronological Matrix Splits Completed", 
                train_rows=len(X_train), val_rows=len(X_val), test_rows=len(X_test), 
                features_count=len(config_ml.FEATURES))
    return X_train, y_train, X_val, y_val, is_holiday_val, X_test, y_test, is_holiday_test

def train_model(model_type: str, params: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a machine learning model pipeline and calculate the WMAPE on the test set.

    Args:
        model_type (str): The name of the regression algorithm to initialize.
        params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
        Tuple[Pipeline, Dict[str, float]]: The fitted model pipeline and a dictionary of evaluation metrics.
    """
    X_train, y_train, X_val, y_val, is_holiday_val, X_test, y_test, is_holiday_test = prepare_data()
    
    logger.info("Training Model Pipeline", model_type=model_type, params=params)
    pipeline = get_model_pipeline(model_type, config_ml.NUMERIC_FEATURES, config_ml.CATEGORICAL_FEATURES, params)
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    wmape = wmape_metric(y_test.values, preds, is_holiday_test)

    return pipeline, {"WMAPE":wmape}


if __name__ == '__main__':
    
    _, metrics = train_model("LinearRegression", {})
    print(f"LinearRegression Pipeline WMAPE: {metrics['WMAPE']}")

    _, metrics = train_model("XGBoost", config_ml.XGBOOST_PARAMS)
    print(f"XGBoost Pipeline WMAPE: {metrics['WMAPE']}")
    