import optuna
import mlflow 
import structlog
import logging
import warnings
import time
import pickle
import pandas as pd
from typing import Dict, Any

import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt
import dagshub
from mlflow.models.signature import infer_signature

from src.utils.config_manager import ConfigManager
from src.data.chronological_split import load_ml_splits
from src.training.metrics import calculate_production_metrics
from src.training.pipeline_factory import get_model_pipeline

# Silence aggressive MLflow/Optuna infrastructure warnings
logging.getLogger("mlflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

logger = structlog.get_logger()

class HyperparameterOptimizer:
    """
    Executes Bayesian Hyperparameter Search using Optuna.
    Tracks all trials, hardware metrics, and performance to MLflow.
    """
    def __init__(self, cfg: ConfigManager):
        self.cfg = cfg
        self.experiment_name = self.cfg.get("training.xgboost.opt_experiment_name")
        self.n_trials = self.cfg.get("training.xgboost.n_trials_optuna", 20)
        
        # Load features dynamically from YAML
        self.numeric_features = self.cfg.get("data.features.numeric")
        self.categorical_features = self.cfg.get("data.features.categorical")
        
        self._setup_tracking()
        self._load_data()

    def _setup_tracking(self):
        """Initializes DagsHub and MLflow."""
        repo_owner, repo_name = self.cfg.get("project.tracking_repo").split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        mlflow.set_experiment(self.experiment_name)

    def _load_data(self):
        """Loads the segregated data splits into class memory."""
        logger.info("Initializing baseline data splits for optimization...")
        self.X_train, self.y_train, self.X_val, self.y_val, self.is_holiday_val, _, _, _ = load_ml_splits()

    def objective(self, trial: optuna.Trial) -> float:
        """The core Bayesian evaluation logic for a single trial."""
        current_depth = trial.suggest_int('max_depth', 3, 10)
        run_name = f"Trial_{trial.number}_Depth{current_depth}"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tags({"Algorithm": "XGBoost", "Phase": "Hyperparameter_Tuning"})
            
            # 1. Define Search Space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': current_depth,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            # Add fixed params from YAML (e.g., random_state, enable_categorical)
            params.update(self.cfg.get("training.xgboost.fixed_params", {}))
            
            mlflow.log_params(params)
            logger.info("Starting Trial...", trial=trial.number)
            
            # 2. Train Pipeline
            pipeline = get_model_pipeline("XGBoost", self.numeric_features, self.categorical_features, params)
            pipeline.fit(self.X_train, self.y_train)
            
            # 3. Evaluate Quality Metrics
            preds_val = pipeline.predict(self.X_val)
            metrics = calculate_production_metrics(self.y_val.values, preds_val, self.is_holiday_val)
            
            # 4. Evaluate Hardware Constraints (Latency & Size)
            start_time = time.perf_counter()
            pipeline.predict(self.X_val.iloc[[0]])
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            model_size_mb = len(pickle.dumps(pipeline)) / (1024 * 1024)
            
            # 5. Log Everything
            metrics.update({"Latency_ms": latency_ms, "Model_Size_MB": model_size_mb})
            mlflow.log_metrics({f"{k}_Val" if k not in ["Latency_ms", "Model_Size_MB"] else k: v for k, v in metrics.items()})
            
            signature = infer_signature(self.X_val, preds_val)
            mlflow.sklearn.log_model(pipeline, "xgboost_pipeline_artifact", signature=signature)
            
            return metrics["WMAPE"]

    def run_optimization(self):
        """Triggers the Optuna study and logs meta-analytics."""
        logger.info("Triggering Bayesian Search", n_trials=self.n_trials)
        study = optuna.create_study(direction='minimize')
        
        # We must use n_jobs=1 to respect DagsHub's free tier rate limits
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)
        
        logger.info("Optimization Grid Complete!", best_wmape=study.best_trial.value)
        self._log_analytics(study)

    def _log_analytics(self, study: optuna.Study):
        """Generates and uploads Optuna charts to a meta-MLflow run."""
        with mlflow.start_run(run_name="Tuning_Analytics"):
            try:
                fig_imp = vis.plot_param_importances(study)
                plt.tight_layout()
                mlflow.log_figure(fig_imp.figure, "hyperparameter_importance.png")
                plt.close()
                
                fig_hist = vis.plot_optimization_history(study)
                plt.tight_layout()
                mlflow.log_figure(fig_hist.figure, "optimization_history.png")
                plt.close()
                
                df_trials = study.trials_dataframe()
                df_trials.to_csv("optuna_raw_trace.csv", index=False)
                mlflow.log_artifact("optuna_raw_trace.csv")
            except Exception as e:
                logger.warning("Could not generate visual analytics. Usually due to low n_trials.", error=str(e))

if __name__ == '__main__':
    # CI/CD can effortlessly swap params.yaml for test.yaml here!
    cfg = ConfigManager("config/params.yaml")
    optimizer = HyperparameterOptimizer(cfg)
    optimizer.run_optimization()