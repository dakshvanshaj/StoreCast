import pandas as pd
import time
import pickle
import structlog
import warnings
import logging
from typing import Dict, Any, Tuple

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
import dagshub

from src.utils.config_manager import ConfigManager
from src.training.metrics import calculate_production_metrics
from src.data.chronological_split import load_ml_splits
from src.training.pipeline_factory import get_model_pipeline

logger = structlog.get_logger()
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

class ModelDeployer:
    """
    Enterprise Model Deployment Pipeline.
    Responsible for identifying the optimal candidate from experimentation,
    retraining on full historical data, and evaluating against production quality gates.
    """
    
    def __init__(self, tracking_repo: str, opt_experiment: str, deploy_experiment: str, quality_gates: Dict[str, float]):
        self.opt_experiment = opt_experiment
        self.deploy_experiment = deploy_experiment
        self.quality_gates = quality_gates
        
        logger.info("Initializing DagsHub connection...")
        repo_owner, repo_name = tracking_repo.split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        self.client = MlflowClient()

    def find_best_candidate(self) -> Dict[str, Any]:
        """Queries MLflow for the best run using a Multi-Objective Pareto Sort."""
        logger.info("Querying MLflow Registry for top candidate pipelines...")
        experiment = self.client.get_experiment_by_name(self.opt_experiment)
        
        if experiment is None:
            raise ValueError(f"Experiment '{self.opt_experiment}' not found.")

        # Pre-filter runs based on the WMAPE quality gate
        max_wmape = self.quality_gates.get("max_wmape", 8.5)
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.WMAPE_Val < {max_wmape}",
            max_results=10
        )

        if not runs:
            raise RuntimeError(f"CRITICAL: No optimization traces met the WMAPE < {max_wmape} tolerance!")

        # Multi-Objective Pareto Sorting
        def calculate_production_score(run):
            m = run.data.metrics
            return (m.get('WMAPE_Val', 10.0) * 0.70) + (m.get('Latency_ms', 50.0) * 0.20) + (m.get('Model_Size_MB', 5.0) * 0.10)
            
        best_run = sorted(runs, key=calculate_production_score)[0]
        
        logger.info(
            "🏆 CHAMPION CANDIDATE FOUND! 🏆", 
            run_name=best_run.info.run_name, 
            pareto_score=round(calculate_production_score(best_run), 2)
        )
        
        # Cast MLflow string params back to native types
        raw_params = best_run.data.params
        return {
            'n_estimators': int(raw_params['n_estimators']),
            'learning_rate': float(raw_params['learning_rate']),
            'max_depth': int(raw_params['max_depth']),
            'subsample': float(raw_params['subsample']),
            'colsample_bytree': float(raw_params['colsample_bytree'])
        }

    def train_production_model(self, best_params: Dict[str, Any]) -> Tuple[Pipeline, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Any]:
        """Retrains the model using Train + Val data."""
        logger.info("Loading chronologically segregated matrices...")
        X_train, y_train, X_val, y_val, _, X_test, y_test, is_holiday_test = load_ml_splits()
        
        logger.info("Concatenating Historical Data (Train + Val) for mathematical maximization...")
        X_train_full = pd.concat([X_train, X_val], axis=0)
        y_train_full = pd.concat([y_train, y_val], axis=0)
        
        # Recast Pandas categorical schemas to prevent XGBoost C-pointer crashes
        cfg = ConfigManager()
        for col in cfg.get("data.features.categorical"):
            X_train_full[col] = X_train_full[col].astype('category')
            
        logger.info("Training unified Booster Pipeline...", params=best_params)
        pipeline_prod = get_model_pipeline(
            "XGBoost", cfg.get("data.features.numeric"), cfg.get("data.features.categorical"), best_params
        )
        pipeline_prod.fit(X_train_full, y_train_full)
        
        return pipeline_prod, X_test, y_test, is_holiday_test

    def evaluate_and_promote(self, pipeline_prod: Pipeline, best_params: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, is_holiday_test: Any) -> None:
        """Evaluates the model against un-seen data and promotes it to the Registry if it passes gates."""
        mlflow.set_experiment(self.deploy_experiment)
        
        with mlflow.start_run(run_name="Champion_Deployment") as run:
            mlflow.set_tags({"Algorithm": "XGBoost", "Phase": "Production_Deployment"})
            mlflow.log_params(best_params)
            
            logger.info("Evaluating Terminal Architecture against Unseen Test Constraints...")
            preds_test = pipeline_prod.predict(X_test)
            metrics_test = calculate_production_metrics(y_test.values, preds_test, is_holiday_test)
            
            # Calculate True Production Latency & Size
            start_time = time.perf_counter()
            pipeline_prod.predict(X_test.iloc[[0]])
            true_latency_ms = (time.perf_counter() - start_time) * 1000.0
            true_size_mb = len(pickle.dumps(pipeline_prod)) / (1024 * 1024)
            
            metrics_test.update({"Latency_ms": true_latency_ms, "Model_Size_MB": true_size_mb})
            mlflow.log_metrics({f"{k}_Test" if k not in ["Latency_ms", "Model_Size_MB"] else k: v for k, v in metrics_test.items()})
            
            signature = infer_signature(X_test, preds_test)
            mlflow.sklearn.log_model(pipeline_prod, "xgboost_pipeline_artifact", signature=signature)
            
            self._check_gates_and_register(run.info.run_id, metrics_test)

    def _check_gates_and_register(self, run_id: str, metrics: Dict[str, float]) -> None:
        """Internal method to validate metrics against quality gates and execute the S3 upload."""
        wmape_pass = metrics["WMAPE"] <= self.quality_gates.get("max_wmape", 8.5)
        r2_pass = metrics["R2"] >= self.quality_gates.get("min_r2", 0.95)
        latency_pass = metrics["Latency_ms"] <= self.quality_gates.get("max_latency_ms", 25.0)

        if wmape_pass and r2_pass and latency_pass:
            logger.info("Evaluating Final Matrix against Enterprise Quality Gates: PASSED.")
            model_uri = f"runs:/{run_id}/xgboost_pipeline_artifact"
            
            # Robust CI/CD Retry Logic for S3 Async Uploads
            for attempt in range(5):
                try:
                    reg_model = mlflow.register_model(model_uri, "StoreCast_XGBoost")
                    self.client.set_registered_model_alias("StoreCast_XGBoost", "production", reg_model.version)
                    logger.info("🏆 Champion Tagged & Promoted!", version=reg_model.version, alias="production", wmape=metrics["WMAPE"])
                    break
                except Exception as e:
                    if attempt < 4:
                        logger.warning("DagsHub S3 async upload delay. Retrying registry promotion in 10s...", attempt=attempt+1)
                        time.sleep(10)
                    else:
                        logger.error("DagsHub Registry Communication Error after retries!", error=str(e))
                        raise
        else:
            logger.warning("CRITICAL: Final Model failed Quality Gates! Deployment BLOCKED.", 
                           wmape=metrics["WMAPE"], r2=metrics["R2"], latency_ms=metrics["Latency_ms"])
            raise RuntimeError("Model did not pass production quality gates.")

if __name__ == '__main__':
    # Define parameters here instead of deep inside the functions.
    # In Phase 2, an orchestrator (like Airflow) will pass these in from a YAML file!
    cfg = ConfigManager()
    config_params = {
        "tracking_repo": cfg.get("project.tracking_repo"),
        "opt_experiment": cfg.get("training.xgboost.opt_experiment_name"),
        "deploy_experiment": cfg.get("training.xgboost.deploy_experiment_name"),
        "quality_gates": {
            "max_wmape": cfg.get("deployment.quality_gates.max_wmape"),
            "min_r2": cfg.get("deployment.quality_gates.min_r2"),
            "max_latency_ms": cfg.get("deployment.quality_gates.max_latency_ms")
        }
    }
    
    deployer = ModelDeployer(**config_params)
    try:
        best_hyperparameters = deployer.find_best_candidate()
        model_pipeline, test_x, test_y, test_holidays = deployer.train_production_model(best_hyperparameters)
        deployer.evaluate_and_promote(model_pipeline, best_hyperparameters, test_x, test_y, test_holidays)
    except Exception as e:
        logger.error("Deployment Pipeline Failed", error=str(e))
        exit(1)