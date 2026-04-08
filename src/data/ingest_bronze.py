from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
import structlog

import config

logger = structlog.get_logger()

def create_bronze_layer() -> None :
    logger.info("bronze_ingestion_started", datasets=["sales", "features", "stores"])
    
    # 1. Delta Lake Configuration Builder
    builder = SparkSession.builder.appName("StoreCast_Bronze") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    try:
        # 2. Read
        logger.info("reading_raw_csvs")
        sales = spark.read.csv(str(config.RAW_SALES_PATH), header=True, inferSchema=True)
        features = spark.read.csv(str(config.RAW_FEATURES_PATH), header=True, inferSchema=True)
        stores = spark.read.csv(str(config.RAW_STORES_PATH), header=True, inferSchema=True)

        # 3. Write Delta
        logger.info("writing_delta_partitions", dataset="sales")
        sales.write.format("delta").partitionBy("Store").mode("overwrite").save(str(config.BRONZE_SALES_PATH))
        
        logger.info("writing_delta_tables", datasets=["features", "stores"])
        features.write.format("delta").mode("overwrite").save(str(config.BRONZE_FEATURES_PATH))
        stores.write.format("delta").mode("overwrite").save(str(config.BRONZE_STORES_PATH))

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