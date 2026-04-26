import os
import shap
import mlflow
import dagshub
import structlog
import logging
import warnings
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from src.data.chronological_split import load_ml_splits

logger = structlog.get_logger()

logging.getLogger("mlflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

def export_shap_analytics():
    logger.info("Initializing Explainer Pipeline...")
    
    from src.deployment.load_champion import download_champion_model
    pipeline = download_champion_model()
    if pipeline is None:
        return
    
    # 3. Extract the bare-metal XGBoost Regressor from inside the Pipeline
    # Our architecture: get_model_pipeline returns Pipeline(steps=[('model', TransformedTargetRegressor())])
    logger.info("Unwrapping internal Scikit-Learn TransformedTargetRegressor...")
    raw_xgb_model = pipeline.named_steps['model'].regressor_
    
    # 4. Fetch the data to explain
    _, _, _, _, _, X_test, _, _ = load_ml_splits()
    
    # Down-sample to 5,000 rows to protect memory. 
    # SHAP calculates Tree Explanations on X_val to show how the model evaluates UNSEEN temporal contexts
    sample_size = min(5000, len(X_test))
    X_explainer_sample = X_test.sample(n=sample_size, random_state=42)
    
    logger.info("Ripping decision tree geometries for Game-Theoretic Analysis...", sample_size=sample_size)
    
    # 5. Execute SHAP Mathematics
    explainer = shap.TreeExplainer(raw_xgb_model)
    shap_values = explainer(X_explainer_sample)
    
    # Hack to ensure feature names aren't dropped by the XGBoost C-API during extraction
    shap_values.feature_names = list(X_explainer_sample.columns)
    
    # Ensure SHAP export directory exists
    export_dir = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images", "shap")
    os.makedirs(export_dir, exist_ok=True)

    # 6. Generate Global Feature Independence Visualization
    logger.info("Generating Detailed Summary Plot...")
    plt.figure(figsize=(12, 10))
    # CRITICAL: Overriding max_display so Store Type and missing cyclical features aren't truncated!
    shap.summary_plot(shap_values, X_explainer_sample, max_display=len(X_explainer_sample.columns), show=False)
    
    summary_path = os.path.join(export_dir, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Generate a Dependence Plot (2D Interaction)
    logger.info("Generating SHAP Dependence Plot for lag_52...")
    plt.figure(figsize=(10, 8))
    # SHAP automatically pairs lag_52 with its mathematically closest interacting feature
    shap.dependence_plot("lag_52", shap_values.values, X_explainer_sample, show=False)
    dep_path = os.path.join(export_dir, "shap_dependence.png")
    plt.tight_layout()
    plt.savefig(dep_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 8. Local Business Level Explanation (A single store/department's prediction)
    try:
        # Instead of risking the random 5,000 row sample containing 0 holidays,
        # we pull directly from the original Training set. This bypasses pd.concat
        # which silently destroys pandas Categorical dtypes and converts them to 'object'.
        X_train, _, _, _, _, _, _, _ = load_ml_splits()
        
        holiday_mask = X_train['isholiday'].astype(str).isin(['1', '1.0', 'True', 'true'])
        
        holiday_row = X_train[holiday_mask].iloc[[0]]
        normal_row = X_train[~holiday_mask].iloc[[0]]
        
        # Explainer runs instantly on 2 targeted rows
        holiday_shap = explainer(holiday_row)
        normal_shap = explainer(normal_row)
        
        logger.info("Generating Local Waterfall Plot for a HOLIDAY week...")
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(holiday_shap[0], show=False)
        waterfall_holiday_path = os.path.join(export_dir, "shap_waterfall_holiday.png")
        plt.tight_layout()
        plt.savefig(waterfall_holiday_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Generating Local Waterfall Plot for an ORDINARY week...")
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(normal_shap[0], show=False)
        waterfall_normal_path = os.path.join(export_dir, "shap_waterfall_ordinary.png")
        plt.tight_layout()
        plt.savefig(waterfall_normal_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error("Failed to parse Holiday/Normal split.", error=str(e))
        
    # 9. Extract Native XGBoost Feature Importance (Gain)
    try:
        logger.info("Generating XGBoost Native Feature Importance (Information Gain)...")
        
        # We explicitly map the feature names back into the Booster so the plot is readable
        booster = raw_xgb_model.get_booster()
        booster.feature_names = list(X_train.columns)
        
        plt.figure(figsize=(12, 10))
        # xgb API natively binds to plt, does not accept 'show' argument.
        ax = xgb.plot_importance(
            booster, 
            importance_type='gain', 
            max_num_features=len(X_train.columns),
            title="XGBoost Native Feature Importance (By Information Gain)"
        )
        # Resize the plot dynamically
        fig = ax.figure
        fig.set_size_inches(12, 10)
        plt.tight_layout()
        
        native_path = os.path.join(export_dir, "xgb_native_importance.png")
        plt.savefig(native_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.error("Failed to extract Native XGBoost Importances.", error=str(e))
    
    logger.info("Successfully exported all Analytics Artifacts (Summary, Dependence, Waterfall, Native) to docs/images/shap/!")

if __name__ == "__main__":
    export_shap_analytics()
