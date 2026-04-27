import polars as pl
import polars.selectors as cs
import pandas as pd
import mlflow
import structlog
import dagshub
from pathlib import Path

from src.utils.config_manager import ConfigManager

logger = structlog.get_logger()

class BatchInferencer:
    """
    Production Inference Service.
    Retrieves the officially sanctioned @production model and scores the latest features.
    """
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.tracking_repo = self.cfg.get("project.tracking_repo")
        self.model_name = "StoreCast_XGBoost"
        
        self.gold_data_path = self.cfg.get("data.paths.gold_data")
        self.export_path = Path(self.cfg.get("data.paths.predictions_export"))
        
        # Combine lists dynamically
        self.numeric_features = self.cfg.get("data.features.numeric")
        self.categorical_features = self.cfg.get("data.features.categorical")
        self.all_features = self.numeric_features + self.categorical_features
        self.target_col = self.cfg.get("data.columns.target")
        
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Authenticates with DagsHub MLflow registry."""
        repo_owner, repo_name = self.tracking_repo.split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    def load_champion_model(self):
        """Fetches the active production model over the network."""
        model_uri = f"models:/{self.model_name}@production"
        logger.info("Downloading Cloud Registry Champion Model...", uri=model_uri)
        try:
            self.pipeline = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error("Failed to download model! Ensure a @production model exists.", error=str(e))
            raise RuntimeError("Inference halted: No Production Model available.") from e

    def load_online_features(self) -> pd.DataFrame:
        """
        Extracts the latest un-scored data window from the Gold layer for prediction.
        """
        logger.info("Connecting to Gold data layer...")
        lf = pl.scan_parquet(self.gold_data_path).drop_nulls()
        
        # Enforce strict statistical types to align perfectly with the Scikit-Learn training schema
        lf = lf.with_columns(cs.integer().cast(pl.Float64))
        
        for col in self.categorical_features:
            lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
            
        # Isolate the future prediction window (e.g., the upcoming week)
        df_live = lf.tail(1000).collect().to_pandas()
        return df_live

    def execute_batch(self):
            """Orchestrates the end-to-end inference run."""
            self.load_champion_model()
            df_live = self.load_online_features()
            
            # === THE ROBUST FIX ===
            # Ask the model exactly what columns it needs, in what order
            expected_cols = list(self.pipeline.feature_names_in_)
            X_live = df_live[expected_cols] 
            
            logger.info("Executing Production Inference Run...", rows=len(X_live))
            
            # Generate Forecasts
            predictions = self.pipeline.predict(X_live)
            df_live['Predicted_Weekly_Sales'] = predictions
            
            # Secure I/O mapping using global configuration paths
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Slice strictly to required business dashboard KPIs
            business_view = df_live[['store', 'dept', 'date', self.target_col, 'Predicted_Weekly_Sales']]
            business_view.to_csv(self.export_path, index=False)
            
            logger.info("Successfully exported inference artifact!", path=str(self.export_path))
            
if __name__ == "__main__":
    # In production, the orchestrator triggers this exact block
    cfg = ConfigManager("config/params.yaml")
    inferencer = BatchInferencer(cfg)
    try:
        inferencer.execute_batch()
    except Exception as err:
        logger.error("Batch Inference Job Failed!", error=str(err))
        exit(1)