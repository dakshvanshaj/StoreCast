
import os
import structlog
import polars as pl
import polars.selectors as cs

from src.utils.config_manager import ConfigManager

logger = structlog.get_logger()

def generate_splits() -> None:
    """
    Executes the chronological split and writes the datasets to disk.
    """
    logger.info("Generating static chronological datasets...")
    
    cfg = ConfigManager()
    
    # 1. Ensure output directory exists
    ml_data_dir = os.path.dirname(cfg.get("data.paths.ml_train"))
    os.makedirs(ml_data_dir, exist_ok=True)
    
    # 2. Load Gold Data
    logger.info("Loading Gold layer from disk...", path=str(cfg.get("data.paths.gold_data")))
    lf = pl.scan_parquet(cfg.get("data.paths.gold_data"))
    
    # 3. Apply Mandatory Filters and Sorting
    logger.info("Removing 52-week warm-up period and sorting chronologically...")
    # Polars optimization: Instead of drop_nulls() on everything, we specifically target
    # the exact feature indicating the end of the warm-up period.
    lf = lf.filter(pl.col('sales_last_year').is_not_null()).sort('date')
    
    # Cast integers to floats to prevent downstream schema enforce crashes (like BentoML/MLflow)
    lf = lf.with_columns(cs.integer().cast(pl.Float64))
    
    # Cast categorical string types strictly for Pandas downstream
    for col in cfg.get("data.features.categorical"):
        lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
    
    # 4. Calculate Split Boundaries
    logger.info("Calculating chronological split boundaries...")
    split_date_1 = lf.select(pl.col('date').quantile(cfg.get("data.splits.train_quantile"))).collect().item()
    split_date_2 = lf.select(pl.col('date').quantile(cfg.get("data.splits.val_quantile"))).collect().item()
    
    # 5. Execute Splits
    logger.info("Materializing DataFrame splits into memory...")
    train_df = lf.filter(pl.col('date') <= split_date_1).collect().to_pandas()
    val_df = lf.filter((pl.col('date') > split_date_1) & (pl.col('date') <= split_date_2)).collect().to_pandas()
    test_df = lf.filter(pl.col('date') > split_date_2).collect().to_pandas()
    
    logger.info(
        "Split execution successful.", 
        train_rows=len(train_df), 
        val_rows=len(val_df), 
        test_rows=len(test_df)
    )
    
    # 6. Save Artifacts
    logger.info("Writing ML datasets to disk...", path=str(ml_data_dir))
    train_df.to_parquet(cfg.get("data.paths.ml_train"), index=False)
    val_df.to_parquet(cfg.get("data.paths.ml_val"), index=False)
    test_df.to_parquet(cfg.get("data.paths.ml_test"), index=False)
    
    logger.info("Chronological split complete!")

from typing import Any

def load_ml_splits() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Loads the statically materialized chronological datasets.

    Returns:
        tuple: Contains the following elements in order:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training target.
            - X_val (pd.DataFrame): Validation features.
            - y_val (pd.Series): Validation target.
            - is_holiday_val (np.ndarray): Validation holiday boolean array.
            - X_test (pd.DataFrame): Test features.
            - y_test (pd.Series): Test target.
            - is_holiday_test (np.ndarray): Test holiday boolean array.
    """
    logger.info("Loading static datasets from disk...")
    cfg = ConfigManager()
    
    try:
        import pandas as pd
        train_df = pd.read_parquet(cfg.get("data.paths.ml_train"))
        val_df = pd.read_parquet(cfg.get("data.paths.ml_val"))
        test_df = pd.read_parquet(cfg.get("data.paths.ml_test"))
    except FileNotFoundError:
        logger.error("Static ML datasets not found. You MUST run `uv run python -m src.data.chronological_split` first!")
        raise
        

    features = cfg.get("data.features.numeric") + cfg.get("data.features.categorical") + cfg.get("data.features.passthrough")
    target = cfg.get("data.columns.target")
    holiday_col = cfg.get("data.columns.holiday")

    X_train = train_df[features]
    y_train = train_df[target]

    X_val = val_df[features]
    y_val = val_df[target]
    is_holiday_val = val_df[holiday_col].values

    X_test = test_df[features]
    y_test = test_df[target]
    is_holiday_test = test_df[holiday_col].values

    logger.info("ML Data successfully loaded into memory.", 
                train_rows=len(X_train), val_rows=len(X_val), test_rows=len(X_test), 
                features_count=len(features))
                
    return X_train, y_train, X_val, y_val, is_holiday_val, X_test, y_test, is_holiday_test

if __name__ == '__main__':
    generate_splits()
