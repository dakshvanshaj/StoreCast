import time
import pandas as pd
import structlog
import mlflow
import dagshub
from datetime import datetime
from pathlib import Path

from evidently.ui.workspace import Workspace
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping

from src.utils.config_manager import ConfigManager
from src.data.chronological_split import load_ml_splits

logger = structlog.get_logger()

class ModelMonitor:
    """
    Observability pipeline executing Data & Concept Drift checks.
    Evaluates Reference (Training) data against Current (Live/Test) data.
    """
    def __init__(self, cfg: ConfigManager, include_target: bool = True):
        self.cfg = cfg
        self.include_target = include_target
        
        # Pull configurations from YAML
        self.project_name = self.cfg.get("observability.drift_monitor.project_name", "StoreCast Monitoring")
        self.workspace_path = self.cfg.get("observability.drift_monitor.workspace_path", "workspace")
        self.sample_size = self.cfg.get("observability.explainability.sample_size", 5000)
        
        # Alerting Policies
        self.warn_threshold = self.cfg.get("observability.drift_monitor.thresholds.warning", 0.30)
        self.crit_threshold = self.cfg.get("observability.drift_monitor.thresholds.critical", 0.50)
        
        self.numeric_features = self.cfg.get("data.features.numeric")
        self.categorical_features = self.cfg.get("data.features.categorical")
        
        self._setup_connections()

    def _setup_connections(self):
        """Initializes DagsHub and Evidently Workspace."""
        repo_owner, repo_name = self.cfg.get("project.tracking_repo").split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
        
        Path(self.workspace_path).mkdir(parents=True, exist_ok=True)
        self.ws = Workspace.create(self.workspace_path)

    def _get_or_create_project(self):
        for p in self.ws.search_project(self.project_name):
            if p.name == self.project_name:
                return p
        project = self.ws.create_project(self.project_name)
        project.description = "Observability dashboard tracking Covariate, Prediction, and Target Drift."
        project.save()
        return project

    def fetch_champion(self):
        """Downloads the active production model to generate predictions for the drift report."""
        model_uri = "models:/StoreCast_XGBoost@production"
        logger.info("Downloading Cloud Registry Champion Model...", uri=model_uri)
        try:
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error("Failed to load @production model. Cannot monitor without it.", error=str(e))
            raise RuntimeError("Monitoring halted: Production model missing.") from e

    def prepare_datasets(self, pipeline) -> tuple:
        """Constructs the explicit Reference and Current datasets with predictions."""
        logger.info("Extracting chronological Reference and Current datasets...")
        X_train, y_train, _, _, _, X_test, y_test, _ = load_ml_splits()

        # Deterministic sampling to protect memory
        X_ref = X_train.sample(n=min(self.sample_size, len(X_train)), random_state=42).copy()
        y_ref = y_train.loc[X_ref.index]
        
        X_curr = X_test.sample(n=min(self.sample_size, len(X_test)), random_state=42).copy()
        y_curr = y_test.loc[X_curr.index]

        logger.info("Executing Inference on splits to calculate Prediction Drift...")
        X_ref['prediction'] = pipeline.predict(X_ref)
        X_curr['prediction'] = pipeline.predict(X_curr)
        
        X_ref['target'] = y_ref.values
        X_curr['target'] = y_curr.values
        
        # Cast categoricals to strings for Evidently's internal logic
        for col in self.categorical_features:
            if col in X_ref.columns:
                X_ref[col] = X_ref[col].astype(str)
                X_curr[col] = X_curr[col].astype(str)

        return X_ref, X_curr

    def run_monitoring(self):
        """Executes the full observability pipeline."""
        start_time = time.time()
        pipeline = self.fetch_champion()
        ref_data, curr_data = self.prepare_datasets(pipeline)
        
        logger.info("Applying Explicit Column Mapping...")
        column_mapping = ColumnMapping(
            target='target',
            prediction='prediction',
            numerical_features=self.numeric_features,
            categorical_features=self.categorical_features
        )

        metrics = [DataDriftPreset()]
        if self.include_target:
            metrics.append(TargetDriftPreset())

        logger.info("Generating Evidently AI Report...")
        report = Report(metrics=metrics, timestamp=datetime.now())
        report.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)
        
        project = self._get_or_create_project()
        self.ws.add_report(project.id, report)

        self._evaluate_policy(report, start_time)

    def _evaluate_policy(self, report: Report, start_time: float):
        """Extracts drift share, evaluates against thresholds, and logs to MLflow."""
        drift_result = report.as_dict().get("metrics", [{}])[0].get("result", {})
        drift_share = drift_result.get("share_of_drifted_columns", 0.0)
        n_drifted = drift_result.get("number_of_drifted_columns", 0)

        summary = {
            "drift_share": drift_share,
            "drifted_features": n_drifted,
            "duration_seconds": round(time.time() - start_time, 2)
        }

        # The Policy Engine Trigger
        if drift_share >= self.crit_threshold:
            logger.error("🚨 CRITICAL DRIFT DETECTED! Trigger CI/CD Retraining Pipeline.", **summary)
        elif drift_share >= self.warn_threshold:
            logger.warning("⚠️ Elevated drift detected. Send Slack Alert.", **summary)
        else:
            logger.info("✅ Drift within acceptable bounds.", **summary)

        logger.info("Logging Observability metrics back to MLflow...")
        mlflow.set_experiment("StoreCast_Monitoring")
        with mlflow.start_run(run_name=f"Drift_Job_{datetime.now().strftime('%Y%m%d')}"):
            mlflow.set_tags({"Job_Type": "Observability", "Target_Included": str(self.include_target)})
            mlflow.log_metrics(summary)

        logger.info("Monitoring Complete! Run `uv run evidently ui` to view the dashboard.")

if __name__ == '__main__':
    cfg = ConfigManager("config/params.yaml")
    monitor = ModelMonitor(cfg, include_target=True)
    try:
        monitor.run_monitoring()
    except Exception as err:
        logger.error("Drift Monitoring Job Failed!", error=str(err))
        exit(1)