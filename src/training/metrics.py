import numpy as np
import structlog
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = structlog.get_logger()

def calculate_production_metrics(y_true: Any, y_pred: Any, is_holiday: Any) -> Dict[str, float]:
    """
    Calculate a suite of regression performance metrics (WMAPE, MAE, RMSE, R2).
    
    Args:
        y_true (np.ndarray): Array of actual sales values.
        y_pred (np.ndarray): Array of predicted sales values.
        is_holiday (np.ndarray): Binary array indicating if a week is a holiday (1) or not (0).

    Returns:
        Dict[str, float]: Dictionary mapping WMAPE, RMSE, MAE, and R2 scores.
    """
    logger.debug("Calculating Production Metric Suite", samples=len(y_true))
    weights = np.where(is_holiday == 1, 5, 1)

    wmape = float(np.round((np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * y_true)) * 100.0, 2))
    mae = float(np.round(mean_absolute_error(y_true, y_pred), 2))
    rmse = float(np.round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    r2 = float(np.round(r2_score(y_true, y_pred), 4))
    
    return {
        "WMAPE": wmape,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }
