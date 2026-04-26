import mlflow
import dagshub
import structlog
from typing import Any, Optional

logger = structlog.get_logger()

def download_champion_model(model_uri: str = "models:/StoreCast_XGBoost@production") -> Optional[Any]:
    """
    Fetches the specified model from the DagsHub MLflow Registry.
    
    Args:
        model_uri (str): The MLflow URI pointing to the desired model version or alias.
                         Defaults to "models:/StoreCast_XGBoost@production".
                         
    Returns:
        The loaded Scikit-Learn pipeline/model, or None if the download fails.
    """
    from src.utils.config_manager import ConfigManager
    cfg = ConfigManager()
    repo = cfg.get("project.tracking_repo")
    repo_owner, repo_name = repo.split('/')
    logger.info("Initializing DagsHub connection...")
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    logger.info("Downloading Cloud Registry Champion Model...", uri=model_uri)
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
        logger.info("Champion model successfully loaded from registry!")
        return pipeline
    except Exception as e:
        logger.error("Failed to download model! Ensure optimizer.py successfully registered a model.", error=str(e))
        return None
