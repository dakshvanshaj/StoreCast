from pathlib import Path

# ============ Project Root =============
PROJECT_ROOT = Path(__file__).parent.resolve()

# ============ RAW DATA PATH ============
RAW_SALES_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"
RAW_FEATURES_PATH = PROJECT_ROOT / "data" / "raw" / "features.csv"
RAW_STORES_PATH  = PROJECT_ROOT / "data" / "raw" / "stores.csv"

# ============ BRONZE DATA PATH ============
BRONZE_SALES_PATH = PROJECT_ROOT / "data" / "bronze" / "sales"
BRONZE_FEATURES_PATH = PROJECT_ROOT / "data" / "bronze" / "features"
BRONZE_STORES_PATH = PROJECT_ROOT / "data" / "bronze" / "stores"