from pathlib import Path

# ============ Project Root =============
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# ============ DATA DIRECTORY ============
DATA_DIR = PROJECT_ROOT / "data"

# ============ RAW DATA PATH ============
RAW_SALES_PATH = DATA_DIR / "raw" / "sales.csv"
RAW_FEATURES_PATH = DATA_DIR / "raw" / "features.csv"
RAW_STORES_PATH  = DATA_DIR / "raw" / "stores.csv"

# ============ BRONZE DATA PATH ============
BRONZE_SALES_PATH = DATA_DIR / "bronze" / "sales"
BRONZE_FEATURES_PATH = DATA_DIR / "bronze" / "features"
BRONZE_STORES_PATH = DATA_DIR / "bronze" / "stores"

# ============ SILVER DATA PATH ============
SILVER_SALES_PATH = DATA_DIR / "silver" / "sales"
SILVER_FEATURES_PATH = DATA_DIR / "silver" / "features"
SILVER_STORES_PATH = DATA_DIR / "silver" / "stores"

# ============ GOLD DATA PATH ============
GOLD_MASTER_PATH = DATA_DIR / "gold" / "master_sales_v1.parquet"

GOLD_DATA_PATH = DATA_DIR / "gold" / "gold_sales.parquet"

# ============ STATIC ML FEATURE STORE ============
ML_DATA_DIR = DATA_DIR / "ml_data"
TRAIN_DATA_PATH = ML_DATA_DIR / "train.parquet"
VAL_DATA_PATH = ML_DATA_DIR / "val.parquet"
TEST_DATA_PATH = ML_DATA_DIR / "test.parquet"

# ============ PREDICTIONS PATH ============
PREDICTIONS_DIR = DATA_DIR / "04_predictions"
BATCH_FORECAST_RESULTS_PATH = PREDICTIONS_DIR / "batch_forecast_results.csv"

# ============ ADVANCED ANALYTICS PATH ============
ADVANCED_ANALYTICS_DIR = DATA_DIR / "05_advanced_analytics"
MARKET_BASKET_RULES_PATH = ADVANCED_ANALYTICS_DIR / "market_basket_rules.csv"
ANOMALIES_PATH = ADVANCED_ANALYTICS_DIR / "high_severity_anomalies.csv"