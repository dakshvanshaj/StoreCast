import yaml
import structlog
from pathlib import Path

logger = structlog.get_logger()

class ConfigManager:
    """
    Centralized Configuration Loader.
    Reads YAML files and exposes them as a dictionary for the application.
    """
    def __init__(self, config_path: str = "config/params.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.error(f"Configuration file not found at {self.config_path}!")
            raise FileNotFoundError(f"Missing config: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            try:
                config_dict = yaml.safe_load(file)
                logger.debug("Configuration loaded successfully.", path=str(self.config_path))
                return config_dict
            except yaml.YAMLError as exc:
                logger.error("Failed to parse YAML configuration!", error=str(exc))
                raise

    def get(self, key_path: str, default=None):
        """
        Allows fetching nested keys using dot notation.
        Example: config.get('training.xgboost.n_trials_optuna')
        """
        keys = key_path.split('.')
        val = self.config
        try:
            for key in keys:
                val = val[key]
            return val
        except KeyError:
            logger.warning(f"Config key '{key_path}' not found. Returning default: {default}")
            return default