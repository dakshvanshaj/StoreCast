import pandas as pd
import time
import pickle
import structlog
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import dagshub
from src.utils.config_manager import ConfigManager
from src.training.metrics import calculate_production_metrics
from src.data.chronological_split import load_ml_splits
from src.training.pipeline_factory import get_model_pipeline

logger = structlog.get_logger()

class CandidateStager:
    """Automated pipeline to train and stage the best model candidate."""
    
    def __init__(self, tracking_repo: str, opt_experiment: str):
        self.opt_experiment = opt_experiment
        repo_owner, repo_name = tracking_repo.split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        self.client = MlflowClient()

    def get_best_hyperparameters(self) -> dict:
            """Finds the best Optuna run using Multi-Objective Pareto Sorting."""
            logger.info("Querying MLflow Registry for top candidate pipelines...")
            experiment = self.client.get_experiment_by_name(self.opt_experiment)
            
            # 1. Pre-filter: Only fetch runs that meet a baseline accuracy threshold
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="metrics.WMAPE_Val < 8.5",
                max_results=10
            )
            
            if not runs:
                raise RuntimeError("CRITICAL: No optimization traces met the baseline WMAPE < 8.5 tolerance!")
                
            # 2. Multi-Objective Pareto Sorting (Lower score is better)
            def calculate_production_score(run):
                m = run.data.metrics
                # Weights: 70% Accuracy, 20% Speed, 10% Size
                return (m.get('WMAPE_Val', 10.0) * 0.70) + (m.get('Latency_ms', 50.0) * 0.20) + (m.get('Model_Size_MB', 5.0) * 0.10)
                
            # 3. Sort and select the most balanced model
            runs_sorted = sorted(runs, key=calculate_production_score)
            best_run = runs_sorted[0]
            
            logger.info(
                "🏆 INTELLIGENT CANDIDATE FOUND! 🏆", 
                run_name=best_run.info.run_name, 
                wmape_val=best_run.data.metrics.get('WMAPE_Val', 'Unknown'),
                latency_ms=best_run.data.metrics.get('Latency_ms', 'Unknown'),
                pareto_score=round(calculate_production_score(best_run), 2)
            )
            
            # 4. Extract and cast parameters
            raw_params = best_run.data.params
            return {
                'n_estimators': int(raw_params['n_estimators']),
                'learning_rate': float(raw_params['learning_rate']),
                'max_depth': int(raw_params['max_depth']),
                'subsample': float(raw_params['subsample']),
                'colsample_bytree': float(raw_params['colsample_bytree'])
            }

    def train_and_stage(self, params: dict):
        """Retrains on full data and tags as @candidate."""
        X_train, y_train, X_val, y_val, _, X_test, y_test, is_holiday = load_ml_splits()
        
        # Combine Train & Val for maximum learning
        X_full = pd.concat([X_train, X_val], axis=0)
        y_full = pd.concat([y_train, y_val], axis=0)
        cfg = ConfigManager()
        for col in cfg.get("data.features.categorical"):
            X_full[col] = X_full[col].astype('category')

        logger.info("Training candidate model...")
        pipeline = get_model_pipeline("XGBoost", cfg.get("data.features.numeric"), cfg.get("data.features.categorical"), params)
        pipeline.fit(X_full, y_full)

        logger.info("Evaluating on unseen Test set...")
        preds = pipeline.predict(X_test)
        metrics = calculate_production_metrics(y_test.values, preds, is_holiday)
        
        # Log to MLflow and register as Candidate
        mlflow.set_experiment("StoreCast_Candidate_Staging")
        with mlflow.start_run(run_name="Candidate_Build"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            
            signature = infer_signature(X_test, preds)
            model_info = mlflow.sklearn.log_model(pipeline, "model", signature=signature)
            
            logger.info("Registering model to Cloud...")
            reg_model = mlflow.register_model(model_info.model_uri, "StoreCast_XGBoost")
            
            # CRITICAL HITL CHANGE: Tag as 'candidate', NOT 'production'
            self.client.set_registered_model_alias("StoreCast_XGBoost", "candidate", reg_model.version)
            logger.info(f"Model successfully staged as @candidate (Version {reg_model.version})", wmape=metrics['WMAPE'])

if __name__ == '__main__':
    cfg = ConfigManager()
    stager = CandidateStager(
        tracking_repo=cfg.get("project.tracking_repo"),
        opt_experiment=cfg.get("training.xgboost.opt_experiment_name")
    )
    best_params = stager.get_best_hyperparameters()
    stager.train_and_stage(best_params)