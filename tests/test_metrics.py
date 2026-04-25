import numpy as np
import pytest
from src.models.trainer import calculate_production_metrics

def test_wmape_holiday_weighting():
    """
    Test that the WMAPE metric correctly penalizes errors on Holiday weeks by a factor of 5.
    If the error occurs on a holiday week, the penalty should drastically increase the WMAPE 
    compared to the exact same error occurring on a normal week.
    """
    # Scenario A: Miss by 50 units on a NORMAL week, but perfect on a HOLIDAY week.
    y_true_a = np.array([100, 100])
    y_pred_a = np.array([50, 100]) # Missed normal week
    is_holiday_a = np.array([0, 1])
    
    # Math:
    # Error Sum = (abs(100-50) * 1) + (abs(100-100) * 5) = 50 + 0 = 50
    # True Sum = (100 * 1) + (100 * 5) = 100 + 500 = 600
    # WMAPE = (50 / 600) * 100 = 8.33%
    metrics_a = calculate_production_metrics(y_true_a, y_pred_a, is_holiday_a)
    assert metrics_a["WMAPE"] == 8.33
    
    # Scenario B: Miss by 50 units on a HOLIDAY week, but perfect on a NORMAL week.
    y_true_b = np.array([100, 100])
    y_pred_b = np.array([100, 50]) # Missed holiday week
    is_holiday_b = np.array([0, 1])
    
    # Math:
    # Error Sum = (abs(100-100) * 1) + (abs(100-50) * 5) = 0 + 250 = 250
    # True Sum = (100 * 1) + (100 * 5) = 100 + 500 = 600
    # WMAPE = (250 / 600) * 100 = 41.67%
    metrics_b = calculate_production_metrics(y_true_b, y_pred_b, is_holiday_b)
    
    assert metrics_b["WMAPE"] == 41.67
    
    # Verification: The exact same absolute error (50 units) creates a 5x larger WMAPE impact 
    # when it happens on a Holiday.
    assert metrics_b["WMAPE"] > metrics_a["WMAPE"]

def test_perfect_prediction():
    y_true = np.array([10, 20, 30])
    y_pred = np.array([10, 20, 30])
    is_holiday = np.array([0, 0, 0])
    
    metrics = calculate_production_metrics(y_true, y_pred, is_holiday)
    assert metrics["WMAPE"] == 0.0
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
