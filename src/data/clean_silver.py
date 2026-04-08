import polars as pl 
import config 
import time
from structlog import getLogger 

logger = getLogger()

def create_silver_layer() -> None:
    try:
        start_time = time.time()
        logger.info("Starting silver layer transformation", layer="silver")

        logger.info("Reading bronze data", source="bronze")
        sales_df = pl.read_delta(str(config.BRONZE_SALES_PATH))
        features_df = pl.read_delta(str(config.BRONZE_FEATURES_PATH))
        stores_df = pl.read_delta(str(config.BRONZE_STORES_PATH))

        logger.info("Standardizing column names", dataset="sales")
        sales_df = sales_df.rename({col: col.strip().lower().replace(" ", "_") for col in sales_df.columns})
        features_df = features_df.rename({col: col.strip().lower().replace(" ", "_") for col in features_df.columns})
        stores_df = stores_df.rename({col: col.strip().lower().replace(" ", "_") for col in stores_df.columns})

        # Deduplicate overlapping sales records
        logger.info("Deduplicating overlapping Date/Store/Dept records", dataset="sales")
        sales_df = sales_df.unique(subset=['store', 'dept', 'date'], keep='first')

        logger.info("Clipping negative weekly sales to 0", dataset="sales")
        sales_df = sales_df.with_columns(pl.col('weekly_sales').clip(lower_bound=0.0))

        logger.info("Converting markdowns to float | Fill NA with 0 | Clip negative values to 0", dataset="features")
        markdown_cols = ['markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5']
        features_df = features_df.with_columns(
            pl.col(markdown_cols).cast(pl.Float32, strict=False).fill_null(0.0).clip(lower_bound=0.0)
        )

        logger.info("Converting CPI and Unemployment to Float | Fill NA with previous values", dataset="features")
        features_df = features_df.with_columns(
            pl.col(['cpi', 'unemployment']).cast(pl.Float32, strict=False).fill_null(strategy='forward')
        )

        logger.info("Converting Date to Date Type", dataset="sales")
        sales_df = sales_df.with_columns(pl.col('date').str.to_date("%d/%m/%Y", strict=False)).sort('date', descending=False)
        
        logger.info("Converting Date to Date Type", dataset="features")
        features_df = features_df.with_columns(pl.col('date').str.to_date("%d/%m/%Y", strict=False)).sort('date', descending=False)

        # Write to Silver 
        logger.info("Writing to Silver", dataset="sales")
        sales_df.write_delta(str(config.SILVER_SALES_PATH), mode="overwrite")
        
        logger.info("Writing to Silver", dataset="features")
        features_df.write_delta(str(config.SILVER_FEATURES_PATH), mode="overwrite")
        
        logger.info("Writing to Silver", dataset="stores")
        stores_df.write_delta(str(config.SILVER_STORES_PATH), mode="overwrite")

        duration = round(time.time() - start_time, 2)
        logger.info("Silver layer transformation completed", status="success", duration_seconds=duration)

    except Exception as e:
        logger.error("Silver layer transformation failed", error=str(e))
        raise

if __name__ == "__main__":
    create_silver_layer()