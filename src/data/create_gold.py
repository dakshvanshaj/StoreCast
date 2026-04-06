import duckdb
import polars as pl
import structlog 
import config 

logger = structlog.get_logger()


def create_gold_layer():

    logger.info('Initializing duckdb engine')
    
    con = duckdb.connect()
    
    logger.info('Loading silver delta tables into polars')
    sales_df = pl.read_delta(config.SILVER_SALES_PATH)
    features_df = pl.read_delta(config.SILVER_FEATURES_PATH)
    stores_df = pl.read_delta(config.SILVER_STORES_PATH)

    logger.info('Executing Gold Dimensional Model joins')

    gold_sql = f"""
        CREATE OR REPLACE TABLE gold_master AS 
        SELECT 
            s.store,
            s.dept,
            s.date,
            s.weekly_sales,
            s.isholiday,
            st.type as store_type,
            st.size as store_size,
            f.temperature,
            f.fuel_price,
            f.cpi,
            f.unemployment,
            f.markdown1,
            f.markdown2,
            f.markdown3,
            f.markdown4,
            f.markdown5,

            -- Total Markdown 
            (f.markdown1 + f.markdown2 + f.markdown3 + f.markdown4 + f.markdown5) as total_markdown,

            -- Extracted Temporal Features
            MONTH(s.date) as month,
            WEEK(s.date) as week_of_year,

            -- 1. Rolling 4-Week Average Sales (Velocity feature)
            AVG(s.weekly_sales) OVER (
                PARTITION BY s.store, s.dept 
                ORDER BY s.date 
                ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
            ) as rolling_4_wk_sales_avg,

            -- 2. Seasonality Lag (Sales from exactly 52 weeks ago!)
            LAG(s.weekly_sales, 52) OVER (
                PARTITION BY s.store, s.dept 
                ORDER BY s.date
            ) as sales_last_year,
            
            -- 3. Macroeconomic CPI lag (CPI is typically published ~3 months after the reference period)
            --    This creates a 3 month lagged CPI value per store, using the CPI from 3 months ago
            --    to avoid look ahead bias and reflect real world data availability.
            LAG(f.cpi, 12) OVER (
                PARTITION BY s.store 
                ORDER BY s.date
            ) as cpi_lag_3_month

        FROM delta_scan('{config.SILVER_SALES_PATH}') s
        LEFT JOIN delta_scan('{config.SILVER_FEATURES_PATH}') f
            ON s.store = f.store AND s.date = f.date
        LEFT JOIN delta_scan('{config.SILVER_STORES_PATH}') st
            ON s.store = st.store
    """

    con.execute(gold_sql)

    logger.info('Exporting gold table to parquet')
    config.GOLD_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY gold_master TO '{config.GOLD_MASTER_PATH}' (FORMAT PARQUET)")
    con.close()

if __name__ == "__main__":
    create_gold_layer()