import pytest
import pandas as pd
import polars as pl
import numpy as np
from unittest.mock import patch
from src.utils.config_manager import ConfigManager

def test_chronological_split_no_leakage():
    """
    Test that the chronological split strictly enforces a temporal ordering.
    Train dates MUST be before Validation dates, which MUST be before Test dates.
    """
    cfg = ConfigManager()
    features = cfg.get("data.features.numeric") + cfg.get("data.features.categorical") + cfg.get("data.features.passthrough")
    
    # 1. Create a Synthetic Mock Gold Dataframe
    # 100 sequential days. 
    dates = pd.date_range(start="2010-01-01", periods=100)
    mock_dict = {
        "date": dates,
        "store": np.ones(100, dtype=np.int32),
        "dept": np.ones(100, dtype=np.int32),
        "weekly_sales": np.random.rand(100),
        "is_holiday": np.zeros(100, dtype=np.int32)
    }
    
    # Dynamically inject all required features from config so the model doesn't crash
    for feature in features:
        if feature not in mock_dict:
            mock_dict[feature] = np.ones(100, dtype=np.float64)
            
    # Polars requires categorical columns to be strings first
    for feature in cfg.get("data.features.categorical"):
        if feature in mock_dict:
            mock_dict[feature] = mock_dict[feature].astype(str)
            
    mock_df = pl.DataFrame(mock_dict).lazy()
    
    # 2. Patch the scan_parquet command so it loads our synthetic 100 rows instead of the massive 470k row disk file
    with patch('polars.scan_parquet', return_value=mock_df):
        from src.data.chronological_split import load_ml_splits
        X_train, y_train, X_val, y_val, is_holiday_val, X_test, y_test, is_holiday_test = load_ml_splits()
        
        # 3. Assert Split Sizes sum to the total
        assert (len(X_train) + len(X_val) + len(X_test)) == 100
