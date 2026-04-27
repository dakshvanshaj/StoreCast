import structlog
import warnings
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline

from src.utils.config_manager import ConfigManager
from src.data.chronological_split import load_ml_splits
from src.training.metrics import calculate_production_metrics
from src.training.pipeline_factory import get_model_pipeline

logger = structlog.get_logger()

class BaselineTrainer:
    """
    Trains and evaluates standalone models outside of the optimization loop. 
    Useful for establishing baselines or running one-off training jobs.
    """
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.numeric_features = self.cfg.get("data.features.numeric")
        self.categorical_features = self.cfg.get("data.features.categorical")
        
        logger.info("Loading baseline data splits into memory...")
        self.X_train, self.y_train, self.X_val, self.y_val, self.is_holiday_val, self.X_test, self.y_test, self.is_holiday_test = load_ml_splits()

    def train(self, model_type: str, params: Dict[str, Any] = None) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Trains a specified model type and evaluates it on the test set.
        """
        if params is None:
            params = {}
            
        logger.info("Training Model Pipeline", model_type=model_type)
        
        pipeline = get_model_pipeline(
            model_type, 
            self.numeric_features, 
            self.categorical_features, 
            params
        )
        
        pipeline.fit(self.X_train, self.y_train)
        
        logger.info("Evaluating on Unseen Test Set...")
        preds = pipeline.predict(self.X_test)
        metrics = calculate_production_metrics(self.y_test.values, preds, self.is_holiday_test)
        
        return pipeline, metrics

if __name__ == '__main__':
    # Initialize the YAML config
    cfg = ConfigManager("config/params.yaml")
    trainer = BaselineTrainer(cfg)
    
    # 1. Train a simple Linear Regression Baseline
    _, lr_metrics = trainer.train("LinearRegression", {})
    logger.info("Linear Regression Baseline", wmape=lr_metrics['WMAPE'])
    
    # 2. Train the XGBoost Default Model
    xgb_params = cfg.get("training.xgboost.fixed_params", {})
    _, xgb_metrics = trainer.train("XGBoost", xgb_params)
    logger.info("XGBoost Default Result", wmape=xgb_metrics['WMAPE'])