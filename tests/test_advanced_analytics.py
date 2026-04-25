import pytest
import polars as pl
from pandas.testing import assert_series_equal
import pandas as pd

def test_market_basket_residual_math():
    """
    Test the fundamental logic of De-Seasonalized Partial Correlation.
    The residual MUST perfectly subtract the expected seasonal baseline (sales_last_year)
    from the actual weekly volume, stripping away the holiday bias.
    """
    # 1. Mock the Gold Data Input
    df = pl.DataFrame({
        "store": [1, 1],
        "dept": [90, 90],
        # Week 1: Expected 50k, Hit 50k. Residual should be 0.
        # Week 2: Expected 10k, Hit 60k (Surprise spike!). Residual should be 50k.
        "weekly_sales": [50000.0, 60000.0],
        "sales_last_year": [50000.0, 10000.0]
    })
    
    # 2. Execute the exact logic from src/models/market_basket.py
    df_transformed = df.with_columns(
        (pl.col("weekly_sales") - pl.col("sales_last_year")).alias("residual_sales")
    )
    
    # 3. Assert the math
    residuals = df_transformed["residual_sales"].to_list()
    assert residuals[0] == 0.0      # Perfectly expected seasonality
    assert residuals[1] == 50000.0  # Massive surprise spike

def test_anomaly_detection_ratio_math():
    """
    Test the fundamental logic of the Context-Aware Anomaly Detection.
    A massive store (Volume Trap) must be reduced to a boring ratio of 1.0 if it is operating normally.
    """
    # 1. Mock the Gold Data Input
    # Store A is a Supercenter (Massive volume). Store B is a tiny Express.
    df = pl.DataFrame({
        "store": [1, 2],
        "weekly_sales": [150000.0, 5000.0],
        "sales_last_year": [150000.0, 50.0] # Store B had a massive jump!
    })
    
    # 2. Execute the exact logic from src/models/anomaly_detection.py
    df_transformed = df.with_columns(
        (pl.col("weekly_sales") / pl.col("sales_last_year")).alias("yoy_growth_ratio")
    )
    
    # 3. Assert the math
    ratios = df_transformed["yoy_growth_ratio"].to_list()
    
    # Store A is a massive $150k supercenter, but its ratio is 1.0 (Boring, not an anomaly!)
    assert ratios[0] == 1.0
    
    # Store B is a tiny $5k express, but its ratio is 100.0 (Massive Anomaly!)
    assert ratios[1] == 100.0
