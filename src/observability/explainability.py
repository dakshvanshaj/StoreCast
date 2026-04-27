import os
import shap
import mlflow
import dagshub
import structlog
import warnings
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.config_manager import ConfigManager
from src.data.chronological_split import load_ml_splits

logger = structlog.get_logger()
warnings.filterwarnings("ignore")

class ModelExplainer:
    """
    Generates SHAP Game-Theoretic explanations for an XGBoost model.
    Decoupled to run against either @candidate or @production aliases.
    """
    def __init__(self, cfg: ConfigManager, target_alias: str = "candidate"):
        self.cfg = cfg
        self.target_alias = target_alias
        self.model_name = "StoreCast_XGBoost"
        
        self.sample_size = self.cfg.get("observability.explainability.sample_size", 5000)
        self.export_dir = Path(self.cfg.get("observability.explainability.export_dir", "docs/images/shap"))
        
        self._setup_mlflow()

    def _setup_mlflow(self):
        repo_owner, repo_name = self.cfg.get("project.tracking_repo").split('/')
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    def generate_explanations(self):
        """Downloads the model, executes SHAP mathematics, and exports visualizations."""
        logger.info(f"Initializing Explainer Pipeline for @{self.target_alias} model...")
        
        # 1. Fetch Model
        model_uri = f"models:/{self.model_name}@{self.target_alias}"
        try:
            pipeline = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load @{self.target_alias} model.", error=str(e))
            raise
            
        logger.info("Unwrapping internal Scikit-Learn TransformedTargetRegressor...")
        raw_xgb_model = pipeline.named_steps['model'].regressor_

        # 2. Fetch Data (Immutable Splits)
        logger.info("Loading chronological data matrices...")
        X_train, _, X_val, _, _, _, _, _ = load_ml_splits()

        # 3. Down-sample for Global SHAP (TreeExplainer scales poorly on massive matrices)
        sample_size = min(self.sample_size, len(X_val))
        X_explainer_sample = X_val.sample(n=sample_size, random_state=42)
        
        logger.info("Executing SHAP Mathematics...", sample_size=sample_size)
        explainer = shap.TreeExplainer(raw_xgb_model)
        shap_values = explainer(X_explainer_sample)
        shap_values.feature_names = list(X_explainer_sample.columns)

        self.export_dir.mkdir(parents=True, exist_ok=True)

        # 4. Global Feature Summary Plot
        logger.info("Generating SHAP Summary Plot...")
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_explainer_sample, max_display=len(X_explainer_sample.columns), show=False)
        plt.tight_layout()
        plt.savefig(self.export_dir / f"shap_summary_{self.target_alias}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Local Business Level Explanation (Targeted Training Rows)
        try:
            holiday_mask = X_train['isholiday'].astype(str).isin(['1', '1.0', 'True', 'true'])
            if holiday_mask.any() and (~holiday_mask).any():
                holiday_row = X_train[holiday_mask].iloc[[0]]
                normal_row = X_train[~holiday_mask].iloc[[0]]
                
                logger.info("Generating Local Waterfall Plots...")
                
                plt.figure(figsize=(12, 8))
                shap.waterfall_plot(explainer(holiday_row)[0], show=False)
                plt.tight_layout()
                plt.savefig(self.export_dir / f"shap_waterfall_holiday_{self.target_alias}.png", dpi=300, bbox_inches='tight')
                plt.close()

                plt.figure(figsize=(12, 8))
                shap.waterfall_plot(explainer(normal_row)[0], show=False)
                plt.tight_layout()
                plt.savefig(self.export_dir / f"shap_waterfall_ordinary_{self.target_alias}.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning("Could not generate Local Waterfall plots.", error=str(e))

        logger.info(f"Successfully exported SHAP Analytics to {self.export_dir}!")

if __name__ == "__main__":
    # By default, we explain the candidate so the human can review it before promotion!
    cfg = ConfigManager("config/params.yaml")
    explainer = ModelExplainer(cfg, target_alias="candidate")
    explainer.generate_explanations()