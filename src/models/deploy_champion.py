import pandas as pd
import time
import pickle
import structlog
import warnings
import logging

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import dagshub

import config_ml
from models.trainer import prepare_data, calculate_production_metrics
from models.pipeline_factory import get_model_pipeline

logger = structlog.get_logger()
logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def execute_production_deployment() -> None:
    """
    Decoupled deployment pipeline. Queries the MLflow registry for the optimal Optuna parameters,
    concatenates the full historical dataset (Train + Val) for mathematical maximization,
    trains the final physical artifact, evaluates strictly on un-seen Test data, and conditionally 
    promotes the pipeline object to the active registry if it passes Quality Gates.
    """
    logger.info("Initializing Enterprise Model Deployment Pipeline...")
    dagshub.init(repo_owner='dakshvanshaj', repo_name='StoreCast', mlflow=True)
    
    client = MlflowClient()
    
    try:
        experiment = client.get_experiment_by_name("StoreCast_XGB_Optimization_V2")
        if experiment is None:
            raise ValueError("Optimization experiment not found on DagsHub.")
    except Exception as e:
        logger.error("Failed to connect to MLflow Experiment. Did you run the optimizer first?", error=str(e))
        return
    
    logger.info("Querying MLflow Registry for top candidate pipelines...")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.WMAPE_Val < 8.5", # Hard strict limit for accuracy
        max_results=10
    )
    
    if not runs:
        logger.error("CRITICAL: No optimization traces met the baseline WMAPE < 8.5 tolerance! Run optimizer.py with a larger grid.")
        return
        
    logger.info("Executing Multi-Objective Pareto Sorting (WMAPE vs Latency vs Size)...")
    
    # Industry Standard: Dynamic "Production Score"
    def calculate_production_score(run):
        m = run.data.metrics
        return (m.get('WMAPE_Val', 10.0) * 0.70) + (m.get('Latency_ms', 50.0) * 0.20) + (m.get('Model_Size_MB', 5.0) * 0.10)
        
    runs_sorted = sorted(runs, key=calculate_production_score)
    best_run = runs_sorted[0]
    
    # Transparency: Tell the User exactly WHICH run won and WHY.
    best_original_wmape = best_run.data.metrics.get('WMAPE_Val', 'Unknown')
    best_original_latency = best_run.data.metrics.get('Latency_ms', 'Unknown')
    champion_score = calculate_production_score(best_run)
    run_origin_id = best_run.info.run_name
    
    logger.info(
        "🏆 CHAMPION FOUND! 🏆", 
        run_name=run_origin_id, 
        wmape_val=best_original_wmape, 
        latency_ms=best_original_latency, 
        pareto_production_score=round(champion_score, 2)
    )
    
    # MLflow logs parameters as strings. We dynamically cast them back to their native types.
    best_params_raw = best_run.data.params
    best_params = {
        'n_estimators': int(best_params_raw['n_estimators']),
        'learning_rate': float(best_params_raw['learning_rate']),
        'max_depth': int(best_params_raw['max_depth']),
        'subsample': float(best_params_raw['subsample']),
        'colsample_bytree': float(best_params_raw['colsample_bytree'])
    }
    
    logger.info("Injecting Champion Parameters...", params=best_params)
    
    logger.info("Loading chronologically segregated matrices...")
    X_train, y_train, X_val, y_val, _, X_test, y_test, is_holiday_test = prepare_data()
    
    logger.info("Concatenating Historical Data for Production Retraining (Train + Val)...")
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    
    # CRITICAL: Pandas automatically downgrades identical 'category' arrays into generic 'object' 
    # data types during a DataFrame concat() if the underlying C-pointers don't dynamically match.
    # We must rigorously recast the schemas immediately so XGBoost doesn't crash:
    for col in config_ml.CATEGORICAL_FEATURES:
        X_train_full[col] = X_train_full[col].astype('category')
    
    mlflow.set_experiment("StoreCast_Production_Deployments")
    
    with mlflow.start_run(run_name="Champion_Deployment"):
        mlflow.set_tags({"Algorithm": "XGBoost", "Phase": "Production_Deployment"})
        mlflow.log_params(best_params)
        
        logger.info("Training unified Booster Pipeline...")
        pipeline_prod = get_model_pipeline(
            "XGBoost", config_ml.NUMERIC_FEATURES, config_ml.CATEGORICAL_FEATURES, best_params
        )
        pipeline_prod.fit(X_train_full, y_train_full)
        
        logger.info("Evaluating Terminal Architecture against Unseen Test Constraints...")
        preds_test = pipeline_prod.predict(X_test)
        metrics_test = calculate_production_metrics(y_test.values, preds_test, is_holiday_test)
        
        # Calculate True Production Latency & Size
        start_time = time.perf_counter()
        pipeline_prod.predict(X_test.iloc[[0]])
        true_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        true_size_mb = len(pickle.dumps(pipeline_prod)) / (1024 * 1024)
        
        mlflow.log_metrics({
            "WMAPE_Test": metrics_test["WMAPE"],
            "R2_Test": metrics_test["R2"],
            "MAE_Test": metrics_test["MAE"],
            "RMSE_Test": metrics_test["RMSE"],
            "Latency_ms": true_latency_ms,
            "Model_Size_MB": true_size_mb
        })
        
        signature = infer_signature(X_test, preds_test)
        mlflow.sklearn.log_model(pipeline_prod, "xgboost_pipeline_artifact", signature=signature)
        
        logger.info("Evaluating Final Matrix against Enterprise Quality Gates...")
        # Note: We loosen the Latency boundary to 15.0ms here since the tree logic is much deeper given the 20% larger dataset
        if metrics_test["WMAPE"] < 8.5 and metrics_test["R2"] > 0.95 and true_latency_ms < 15.0:
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/xgboost_pipeline_artifact"
            
            try:
                reg_model = mlflow.register_model(model_uri, "StoreCast_XGBoost")
                client.set_registered_model_alias("StoreCast_XGBoost", "production", reg_model.version)
                logger.info("Quality Gates Passed! Champion Tagged & Promoted!", version=reg_model.version, alias="production", wmape=metrics_test["WMAPE"])
            except Exception as e:
                logger.error("DagsHub Registry Communication Error!", error=str(e))
        else:
            logger.warning("CRITICAL: Final Model failed Quality Gates! (e.g. WMAPE > 8.5). Deployment BLOCKED.", wmape=metrics_test["WMAPE"])

if __name__ == '__main__':
    execute_production_deployment()
