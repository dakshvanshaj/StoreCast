from pathlib import Path
import config 


def test_lakehouse_paths_exist():

    assert config.BRONZE_SALES_PATH.parent.exists(), "Bronze directory should exist"
    assert config.SILVER_SALES_PATH.parent.exists(), "Silver directory should exist"

def test_config_project_root():
    
    assert "storecast" in str(config.PROJECT_ROOT).lower(), "Config should find the project root"
