from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import structlog
from src.utils.config_manager import ConfigManager

logger = structlog.get_logger()

def create_bronze_layer() -> None:
    logger.info("bronze_ingestion_started", datasets=["sales", "features", "stores"])
    cfg = ConfigManager()
    
    # 1. Delta Lake Configuration Builder
    builder = SparkSession.builder.appName("StoreCast_Bronze") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    try:
        # 2. Read
        logger.info("reading_raw_csvs")
        sales = spark.read.csv(cfg.get("data.paths.raw_sales"), header=True, inferSchema=True)
        features = spark.read.csv(cfg.get("data.paths.raw_features"), header=True, inferSchema=True)
        stores = spark.read.csv(cfg.get("data.paths.raw_stores"), header=True, inferSchema=True)

        # 3. Write Delta
        logger.info("writing_delta_partitions", dataset="sales")
        sales.write.format("delta").partitionBy("Store").mode("overwrite").save(cfg.get("data.paths.bronze_sales"))
        
        logger.info("writing_delta_tables", datasets=["features", "stores"])
        features.write.format("delta").mode("overwrite").save(cfg.get("data.paths.bronze_features"))
        stores.write.format("delta").mode("overwrite").save(cfg.get("data.paths.bronze_stores"))

        logger.info("bronze_ingestion_completed", status="success")

    except Exception as e:
        logger.error("bronze_ingestion_failed", error=str(e))
        raise

    finally:
        # 4. Resilient Shutdown
        logger.info("stopping_spark_session")
        spark.stop()

if __name__ == "__main__":
    create_bronze_layer()