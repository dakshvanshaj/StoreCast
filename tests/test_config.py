from pathlib import Path
from src.utils.config_manager import ConfigManager


def test_lakehouse_paths_exist():
    cfg = ConfigManager()
    assert Path(cfg.get("data.paths.bronze_sales")).parent.exists(), "Bronze directory should exist"
    assert Path(cfg.get("data.paths.silver_sales")).parent.exists(), "Silver directory should exist"

def test_config_manager_loads_yaml():
    cfg = ConfigManager()
    assert cfg.get("project.name") == "StoreCast", "ConfigManager should load the project name"
