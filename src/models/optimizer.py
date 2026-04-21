import optuna
import mlflow 
import structlog
import logging
import warnings
import time
import config_ml
import optuna.visualization.matplotlib as vis
import matplotlib.pyplot as plt
import dagshub
import pickle
from mlflow.models.signature import infer_signature
from models.trainer import prepare_data, calculate_production_metrics
from models.pipeline_factory import get_model_pipeline
from mlflow.tracking import MlflowClient

dagshub.init(repo_owner='dakshvanshaj', repo_name='StoreCast', mlflow=True)
# Silence aggressive MLflow infrastructure warnings to keep Optuna logs pristine
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

logger = structlog.get_logger()

def optimize_xgboost(n_trials: int = 20) -> None:
    """
    Run Optuna hyperparameter optimization for the XGBoost model.
    
    This function sets up an MLflow tracking experiment, prepares the train/validation 
    dataset, and triggers a Bayesian search using Optuna to minimize WMAPE.

    Args:
        n_trials (int): Number of Optuna trials to execute. Defaults to 20.
    """
    logger.info("Initializing baseline data splits for optimization...")
    X_train, y_train, X_val, y_val, is_holiday_val, _, _, _ = prepare_data()
    
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("StoreCast_XGB_Optimization_V2")
    
    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for a single trial evaluating XGBoost.

        Args:
            trial (optuna.Trial): The Optuna trial object containing hyperparameter suggestions.

        Returns:
            float: The validation WMAPE score for the trial.
        """
        current_depth = trial.suggest_int('max_depth', 3, 10)
        run_name = f"Trial_{trial.number}_Depth{current_depth}"
        
        with mlflow.start_run(run_name=run_name, nested=True):
            mlflow.set_tags({"Algorithm": "XGBoost", "Phase": "Hyperparameter_Tuning"})
            
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': current_depth,
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            mlflow.log_params(params)
            
            logger.info("Starting Bayesian Trial Training...", trial_number=trial.number, params=params)
            
            pipeline = get_model_pipeline(
                "XGBoost", config_ml.NUMERIC_FEATURES, config_ml.CATEGORICAL_FEATURES, params
            )
            pipeline.fit(X_train, y_train)
            
            preds_val = pipeline.predict(X_val)
            metrics = calculate_production_metrics(y_val.values, preds_val, is_holiday_val)
            
            # API Single-Row Inference Latency Benchmark
            # Proving to the business our endpoints won't crash under API traffic
            single_row = X_val.iloc[[0]]
            start_time = time.perf_counter()
            pipeline.predict(single_row)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Batch Throughput Benchmark (Rows Per Second)
            # Determines how fast background cron-jobs can process millions of data points
            start_time_batch = time.perf_counter()
            pipeline.predict(X_val)
            batch_duration = time.perf_counter() - start_time_batch
            throughput_rps = len(X_val) / batch_duration
            
            # Model Mathematical Memory Footprint (Megabytes)
            # Critical for Kubernetes RAM allocation cold-starts
            
            model_size_mb = len(pickle.dumps(pipeline)) / (1024 * 1024)
            
            mlflow.log_metrics({
                "WMAPE_Val": metrics["WMAPE"],
                "RMSE_Val": metrics["RMSE"],
                "MAE_Val": metrics["MAE"],
                "R2_Val": metrics["R2"],
                "Latency_ms": latency_ms,
                "Throughput_RPS": throughput_rps,
                "Model_Size_MB": model_size_mb
            })
            
            signature = infer_signature(X_val, preds_val)
            mlflow.sklearn.log_model(pipeline, "xgboost_pipeline_artifact", signature=signature)
            
            logger.info("Trial Recorded Successfully", trial_number=trial.number, wmape=metrics["WMAPE"])
            return metrics["WMAPE"]
    
    logger.info("Triggering Bayesian Search", n_trials=n_trials)
    study = optuna.create_study(direction='minimize')
    # CRITICAL: We cannot use n_jobs=-1 with a DagsHub FREE remote tracking server. 
    # Parallelizing 8-16 processes will instantly trigger a 429 (Too Many Requests) API Rate Limit crash.
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    
    logger.info("Optimization Grid Complete!", best_wmape=study.best_trial.value)
    logger.info("Best Final Parameters", params=study.best_trial.params)

    logger.info("Generating Bayesian Trace Analytics...")

    # Create an MLflow run explicitly just to hold our Meta-Analytics!
    with mlflow.start_run(run_name="Tuning_Analytics"):
        try:
            # 1. Feature Importance (fANOVA)
            fig_imp = vis.plot_param_importances(study)
            plt.tight_layout()
            mlflow.log_figure(fig_imp.figure, "hyperparameter_importance.png")
            
            # 2. Optimization History
            fig_hist = vis.plot_optimization_history(study)
            plt.tight_layout()
            mlflow.log_figure(fig_hist.figure, "optimization_history.png")
        except Exception as e:
            logger.warning("Could not generate Bayesian visual analytics. This is normal if n_trials is extremely low.", error=str(e))
            
        try:
            # 3. Save the Raw Database Trace 
            df_trials = study.trials_dataframe()
            df_trials.to_csv("optuna_raw_trace.csv", index=False)
            mlflow.log_artifact("optuna_raw_trace.csv")
        except Exception as e:
            logger.error("Failed to upload raw trace artifact.", error=str(e))

    logger.info("End-to-End Optimization Grid Complete! Next step: Execute models.deploy_champion")

if __name__ == '__main__':
    optimize_xgboost(n_trials=30)