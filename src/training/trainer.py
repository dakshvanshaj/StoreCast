import polars as pl
import pandas as pd
import numpy as np 
from src.utils.config_manager import ConfigManager
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
import structlog 
from src.data.chronological_split import load_ml_splits
from src.training.metrics import calculate_production_metrics

from src.training.pipeline_factory import get_model_pipeline

logger = structlog.get_logger()

def train_model(model_type: str, params: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train a machine learning model pipeline and calculate the WMAPE on the test set.

    Args:
        model_type (str): The name of the regression algorithm to initialize.
        params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
        Tuple[Pipeline, Dict[str, float]]: The fitted model pipeline and a dictionary of evaluation metrics.
    """
    X_train, y_train, X_val, y_val, is_holiday_val, X_test, y_test, is_holiday_test = load_ml_splits()
    
    cfg = ConfigManager()
    logger.info("Training Model Pipeline", model_type=model_type, params=params)
    pipeline = get_model_pipeline(
        model_type, 
        cfg.get("data.features.numeric"), 
        cfg.get("data.features.categorical"), 
        params
    )
    pipeline.fit(X_train, y_train)
    
    preds = pipeline.predict(X_test)
    metrics = calculate_production_metrics(y_test.values, preds, is_holiday_test)

    return pipeline, metrics


if __name__ == '__main__':
    cfg = ConfigManager()
    _, metrics = train_model("LinearRegression", {})
    print(f"LinearRegression Pipeline WMAPE: {metrics['WMAPE']}")

    _, metrics = train_model("XGBoost", cfg.get("training.xgboost.fixed_params"))
    print(f"XGBoost Pipeline WMAPE: {metrics['WMAPE']}")
    