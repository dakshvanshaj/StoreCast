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