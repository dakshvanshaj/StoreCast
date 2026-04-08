import duckdb
import polars as pl
import structlog 
import config 
import time

logger = structlog.get_logger()


def create_gold_layer():
    try:
        start_time = time.time()
        logger.info('Initializing duckdb engine', layer='gold', engine='duckdb')
        
        con = duckdb.connect()
        
        logger.info('Loading DuckDB Delta Extension', layer='gold')
        con.execute("INSTALL delta; LOAD delta;")
        
        logger.info('Executing Joins And Feature Engineering SQL Query using DuckDB', layer='gold', query_type='window_functions')

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
            ORDER BY s.date, s.store, s.dept
        """

        con.execute(gold_sql)

        logger.info('Exporting gold table to parquet', layer='gold')
        config.GOLD_MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"COPY gold_master TO '{config.GOLD_MASTER_PATH}' (FORMAT PARQUET)")
     
        logger.info('Closing DuckDB connection', layer='gold')
        con.close()

        duration = round(time.time() - start_time, 2)
        logger.info('Gold layer transformation completed', layer='gold', status='success', duration_seconds=duration)

    except Exception as e:
        logger.error('Gold layer transformation failed', layer='gold', error=str(e))
        raise

if __name__ == "__main__":
    create_gold_layer()