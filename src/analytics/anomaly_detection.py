import polars as pl
from sklearn.ensemble import IsolationForest
import structlog
from src.utils.config_manager import ConfigManager
from pathlib import Path

logger = structlog.get_logger()

def execute_contextual_anomaly_hunting() -> None:
    """
    Executes a Context-Aware Anomaly Detection pipeline.
    Prevents "Alert Fatigue" by feeding Contextual Variables (like momentum and holidays) 
    into the algorithmic isolation tree so normal spikes are mathematically ignored.
    """
    logger.info("Initializing Context-Aware Anomaly Detection...")
    
    cfg = ConfigManager()
    lf = pl.scan_parquet(cfg.get("data.paths.gold_data"))
    
    # CRITICAL: We must mathematically drop the first 52-weeks of warm-up data!
    # DuckDB exports the initial lags as native Nulls. If we don't drop them, the .fill_null(0) 
    # below will coerce them to 0, and the Isolation Forest will falsely flag Week 53 as a 
    # massive anomaly because it thinks sales jumped from $0 to $50,000!
    lf = lf.drop_nulls(subset=["sales_last_year"])
    
    # 2. Engineer Contextual Ratios (Defeating the "Volume Outlier" Trap)
    # If we feed absolute dollars, the algorithm blindly flags high-volume departments (like Grocery).
    # By converting sales into dimensionless Ratios, we mathematically force the tree to hunt for Decoupling Deviations!
    # (Adding +1 to denominator prevents divide-by-zero crashes on zero-sales closed weeks).
    lf = lf.with_columns([
        (pl.col('weekly_sales') / (pl.col('sales_last_year') + 1)).alias('yoy_growth_ratio'),
        (pl.col('weekly_sales') / (pl.col('rolling_4_wk_sales_avg') + 1)).alias('trend_deviation_ratio'),
        (pl.col('total_markdown') / (pl.col('weekly_sales') + 1)).alias('markdown_intensity_ratio')
    ])
    
    # Engineer the Advanced Decoupling Ratios
    lf = lf.with_columns([
        (pl.col('markdown_intensity_ratio') / (pl.col('yoy_growth_ratio') + 0.01)).alias('markdown_cannibalization'),
        (pl.col('weekly_sales') / (pl.col('store_size'))).alias('footprint_yield'),
        pl.when(pl.col('isholiday').cast(pl.String) == "1")
          .then(pl.col('trend_deviation_ratio'))
          .otherwise(pl.lit(1.0))
          .alias('holiday_miss_severity')
    ])
    
    # Contextual Isolation
    # We feed it the completely dimensionless RATIOS!
    features = [
        'yoy_growth_ratio', 
        'trend_deviation_ratio', 
        'markdown_intensity_ratio',
        'markdown_cannibalization',
        'footprint_yield',
        'holiday_miss_severity'
    ]
    
    logger.info("Extracting Contextual Feature Matrix (Ratios)...", features=features)
    
    # Isolate just the features we need for math and fill nulls efficiently in Rust
    # Then cast to Pandas natively because Scikit-Learn mandates Numpy arrays!
    X_train = lf.select(features).fill_null(0).collect().to_pandas()
    
    logger.info("Training Isolation Forest to hunt dimensional decoupling...", rows=len(X_train))
    # contamination=0.005 means we mandate the algorithm isolates the most extreme 0.5% of the database
    iso_forest = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
    
    # Train and Score
    anomaly_predictions = iso_forest.fit_predict(X_train)
    
    # 3. Re-attach the predictions back to the core Polars DataFrame
    df_results = lf.collect().with_columns(
        pl.Series("Anomaly_Flag", anomaly_predictions)
    )
    
    # Isolation forest outputs -1 for anomalous, 1 for normal
    # We use Polars fast filtering syntax
    anomalies = df_results.filter(pl.col('Anomaly_Flag') == -1)
    
    # Isolate relevant business attributes to show exactly WHY they were flagged
    business_view = anomalies.select([
        'store', 'dept', 'date', 'isholiday', 'weekly_sales', 
        'sales_last_year', 'rolling_4_wk_sales_avg', 
        'yoy_growth_ratio', 'trend_deviation_ratio', 'total_markdown',
        'markdown_intensity_ratio', 'markdown_cannibalization', 
        'footprint_yield', 'holiday_miss_severity'
    ])
    
    # Push to Advanced Analytics Storage using Global Configuration
    anomalies_path = Path(cfg.get("data.paths.anomalies_export"))
    anomalies_path.parent.mkdir(parents=True, exist_ok=True)
    business_view.write_csv(str(anomalies_path))
    
    logger.info("Anomaly hunting protocol achieved!", anomalies_found=len(anomalies), path=str(anomalies_path))

if __name__ == "__main__":
    execute_contextual_anomaly_hunting()
