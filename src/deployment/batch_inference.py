from pathlib import Path
import polars as pl
import mlflow
import structlog
import dagshub

from src.utils.config_manager import ConfigManager

logger = structlog.get_logger()

def execute_batch_inference() -> None:
    """
    Fetches the @production XGBoost model from the DagsHub Registry and executes
    batch inference over the upcoming week's feature matrix. 
    Exports clean forecast artifacts directly for BI Dashboard ingestion.
    """
    logger.info("Initializing Batch Inference Pipeline...")
    
    from src.deployment.load_champion import download_champion_model
    pipeline = download_champion_model()
    if pipeline is None:
        return
        
    cfg = ConfigManager()
    logger.info("Connecting to Gold Feature Store (Inference Context)...")
    lf = pl.scan_parquet(cfg.get("data.paths.gold_data")).drop_nulls()
    
    # Enforce strict statistical types to align perfectly with the Scikit-Learn training schema
    import polars.selectors as cs
    lf = lf.with_columns(cs.integer().cast(pl.Float64))
    
    for col in cfg.get("data.features.categorical"):
        lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
        
    # Isolate the simulated future prediction window
    df_live = lf.tail(1000).collect().to_pandas()
    
    features = cfg.get("data.features.numeric") + cfg.get("data.features.categorical") + cfg.get("data.features.passthrough")
    X_live = df_live[features]
    
    logger.info("Executing Production Inference Run...", rows=len(X_live))
    predictions = pipeline.predict(X_live)
    
    # Map predictions back to original dataframe
    df_live['Predicted_Weekly_Sales'] = predictions
    
    # Secure I/O mapping using global configuration paths
    predictions_export = Path(cfg.get("data.paths.predictions_export"))
    predictions_export.parent.mkdir(parents=True, exist_ok=True)
    
    # Slice strictly to required business dashboard KPIs 
    business_view = df_live[['store', 'dept', 'date', 'weekly_sales', 'Predicted_Weekly_Sales']]
    business_view.to_csv(str(predictions_export), index=False)
    
    logger.info("Successfully exported inference artifact!", path=str(predictions_export))

if __name__ == "__main__":
    execute_batch_inference()
