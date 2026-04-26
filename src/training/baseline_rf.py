import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import time
from structlog import getLogger

from src.data.chronological_split import load_ml_splits
from src.training.metrics import calculate_production_metrics

logger = getLogger()

def run_feasibility_study() -> None:
    """
    Run a minimum viable ML pipeline to establish a predictive baseline.
    
    This function reads the statically decoupled datasets, trains a naive 
    Random Forest regressor, and evaluates it using the production metrics suite.

    Raises:
        Exception: Propagates any error encountered during pipeline execution.
    """
    try:
        start_time = time.time()
        logger.info("Initializing ML Feasibility Study...", model="RandomForest")
        
        from src.utils.config_manager import ConfigManager
        from src.training.pipeline_factory import get_model_pipeline
        
        # Load Static Feature Store Splits
        X_train, y_train, X_val, y_val, _, X_test, y_test, is_holiday_test = load_ml_splits()
        
        # Concatenate train+val to approximate the old 80/20 train/test split for the baseline
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        
        # Train Random Forest Pipeline (Includes Preprocessing for Categoricals)
        logger.info("Training naive Random Forest Pipeline... (this may take a minute)")
        
        params = {'n_estimators': 100, 'max_depth': 15, 'n_jobs': -1}
        cfg = ConfigManager()
        model = get_model_pipeline(
            'RandomForest', 
            numeric_features=cfg.get("data.features.numeric"), 
            categorical_features=cfg.get("data.features.categorical"), 
            model_params=params
        )
        
        model.fit(X_train_full, y_train_full)
        
        # Predict & Evaluate
        preds = model.predict(X_test)
        
        # Calculate Production Metrics
        metrics = calculate_production_metrics(y_test.values, preds, is_holiday_test)
        error_wmape = metrics['WMAPE']
        
        logger.info('Model WMAPE', score=error_wmape)
        logger.info('Target to Beat', manual_heuristic=11.85) # Note WMAPE is 0-100 formatted now
        
        if error_wmape < 11.85:
            logger.info("FEASIBILITY PASSED! We beat the baseline!", wmape=error_wmape, status="success")
        else:
            logger.error("FEASIBILITY FAILED! We did not beat the baseline.", status="failed")

        duration = round(time.time() - start_time, 2)
        logger.info("ML Feasibility Study completed", duration_seconds=duration)

    except Exception as e:
        logger.error("ML Feasibility Study failed", error=str(e))
        raise

if __name__ == "__main__":
    run_feasibility_study()