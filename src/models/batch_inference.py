from pathlib import Path
import polars as pl
import mlflow
import structlog
import dagshub

import config
import config_ml

logger = structlog.get_logger()

def execute_batch_inference() -> None:
    """
    Fetches the @production XGBoost model from the DagsHub Registry and executes
    batch inference over the upcoming week's feature matrix. 
    Exports clean forecast artifacts directly for BI Dashboard ingestion.
    """
    logger.info("Initializing Batch Inference Pipeline...")
    
    dagshub.init(repo_owner='dakshvanshaj', repo_name='StoreCast', mlflow=True)
    model_uri = "models:/StoreCast_XGBoost@production"
    
    logger.info("Downloading Cloud Registry Champion Model...", uri=model_uri)
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error("Failed to download model! Enusre optimizer.py successfully registered a @production model.", error=str(e))
        return
        
    logger.info("Connecting to Gold Feature Store (Inference Context)...")
    lf = pl.scan_parquet(config.GOLD_DATA_PATH).drop_nulls()
    
    # Enforce strict statistical types to align perfectly with the Scikit-Learn training schema
    import polars.selectors as cs
    lf = lf.with_columns(cs.integer().cast(pl.Float64))
    
    for col in config_ml.CATEGORICAL_FEATURES:
        lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
        
    # Isolate the simulated future prediction window
    df_live = lf.tail(1000).collect().to_pandas()
    X_live = df_live[config_ml.FEATURES]
    
    logger.info("Executing Production Inference Run...", rows=len(X_live))
    predictions = pipeline.predict(X_live)
    
    # Map predictions back to original dataframe
    df_live['Predicted_Weekly_Sales'] = predictions
    
    # Secure I/O mapping using global configuration paths
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Slice strictly to required business dashboard KPIs 
    business_view = df_live[['store', 'dept', 'date', 'weekly_sales', 'Predicted_Weekly_Sales']]
    business_view.to_csv(config.BATCH_FORECAST_RESULTS_PATH, index=False)
    
    logger.info("Successfully exported inference artifact!", path=str(config.BATCH_FORECAST_RESULTS_PATH))

if __name__ == "__main__":
    execute_batch_inference()
