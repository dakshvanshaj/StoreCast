
import os
import structlog
import polars as pl
import polars.selectors as cs

from src import config, config_ml

logger = structlog.get_logger()

def generate_static_splits() -> None:
    """
    Executes the chronological split and writes the datasets to disk.
    """
    logger.info("Generating static chronological datasets...")
    
    # 1. Ensure output directory exists
    os.makedirs(config.ML_DATA_DIR, exist_ok=True)
    
    # 2. Load Gold Data
    logger.info("Loading Gold layer from disk...", path=str(config.GOLD_DATA_PATH))
    lf = pl.scan_parquet(config.GOLD_DATA_PATH)
    
    # 3. Apply Mandatory Filters and Sorting
    logger.info("Removing 52-week warm-up period and sorting chronologically...")
    # Polars optimization: Instead of drop_nulls() on everything, we specifically target
    # the exact feature indicating the end of the warm-up period.
    lf = lf.filter(pl.col('sales_last_year').is_not_null()).sort('date')
    
    # Cast integers to floats to prevent downstream schema enforce crashes (like BentoML/MLflow)
    lf = lf.with_columns(cs.integer().cast(pl.Float64))
    
    # Cast categorical string types strictly for Pandas downstream
    for col in config_ml.CATEGORICAL_FEATURES:
        lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
    
    # 4. Calculate Split Boundaries
    logger.info("Calculating chronological split boundaries...")
    split_date_1 = lf.select(pl.col('date').quantile(config_ml.TRAIN_SPLIT_QUANTILE)).collect().item()
    split_date_2 = lf.select(pl.col('date').quantile(config_ml.VAL_SPLIT_QUANTILE)).collect().item()
    
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
    logger.info("Writing ML datasets to disk...", path=str(config.ML_DATA_DIR))
    train_df.to_parquet(config.TRAIN_DATA_PATH, index=False)
    val_df.to_parquet(config.VAL_DATA_PATH, index=False)
    test_df.to_parquet(config.TEST_DATA_PATH, index=False)
    
    logger.info("Chronological split complete!")

if __name__ == '__main__':
    generate_static_splits()
