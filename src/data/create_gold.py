import duckdb
import polars as pl
import structlog 
from src.utils.config_manager import ConfigManager
import os
from pathlib import Path 
import time

logger = structlog.get_logger()


def create_gold_layer():
    try:
        start_time = time.time()
        logger.info('Initializing duckdb engine', layer='gold', engine='duckdb')
        
        con = duckdb.connect()
        
        logger.info('Loading DuckDB Delta Extension', layer='gold')
        con.execute("INSTALL delta; LOAD delta;")
        
        cfg = ConfigManager()
        project_root = Path(__file__).parent.parent.parent.resolve()
        
        logger.info('Enabling DuckDB JSON Profiling', layer='gold')
        con.execute("PRAGMA enable_profiling='json';")
        con.execute(f"PRAGMA profile_output='{project_root}/duckdb_profile.json';")
        
        logger.info('Executing Joins And Feature Engineering SQL Query using DuckDB', layer='gold', query_type='window_functions')

        gold_sql = f"""
            CREATE OR REPLACE TABLE gold_master AS 
            WITH calendar AS (
                SELECT UNNEST(GENERATE_SERIES(
                    (SELECT MIN(date) FROM delta_scan('{cfg.get("data.paths.silver_sales")}')), 
                    (SELECT MAX(date) FROM delta_scan('{cfg.get("data.paths.silver_sales")}')), 
                    INTERVAL 7 DAYS
                )) AS date
            ),
            unique_stores AS (
                SELECT DISTINCT store, dept FROM delta_scan('{cfg.get("data.paths.silver_sales")}')
            ),
            scaffold AS (
                SELECT u.store, u.dept, c.date
                FROM unique_stores u
                CROSS JOIN calendar c
            ),
            merged_data AS (
                SELECT 
                    s.store, s.dept, s.date,
                    COALESCE(sales.weekly_sales, 0.0) AS weekly_sales,
                    COALESCE(sales.isholiday, f.isholiday, 0) AS isholiday,
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
                    (COALESCE(f.markdown1, 0) + COALESCE(f.markdown2, 0) + COALESCE(f.markdown3, 0) + COALESCE(f.markdown4, 0) + COALESCE(f.markdown5, 0)) as total_markdown
                FROM scaffold s
                LEFT JOIN delta_scan('{cfg.get("data.paths.silver_sales")}') sales 
                    ON s.store = sales.store AND s.dept = sales.dept AND s.date = sales.date
                LEFT JOIN delta_scan('{cfg.get("data.paths.silver_features")}') f
                    ON s.store = f.store AND s.date = f.date
                LEFT JOIN delta_scan('{cfg.get("data.paths.silver_stores")}') st
                    ON s.store = st.store
            )
            SELECT 
                *,
                MONTH(date) as month,
                WEEK(date) as week_of_year,

                -- 1. Log Stabilized Target
                LN(weekly_sales + 1) AS sales_log,
                
                -- 2. Cyclical Fourier Features
                SIN(2 * PI() * WEEK(date) / 52) AS sin_week,
                COS(2 * PI() * WEEK(date) / 52) AS cos_week,
                
                -- 3. Point in Time Lags (Log Space for ML Stability)
                LAG(LN(weekly_sales + 1), 1) OVER w_momentum AS lag_1_log,
                LAG(LN(weekly_sales + 1), 5) OVER w_momentum AS lag_5_log,
                LAG(LN(weekly_sales + 1), 52) OVER w_momentum AS lag_52_log,
                
                -- 3b. Point in Time Lags (Raw Dollars for BI Dashboards & Operations)
                LAG(weekly_sales, 1) OVER w_momentum AS lag_1_sales,
                LAG(weekly_sales, 5) OVER w_momentum AS lag_5_sales,
                LAG(weekly_sales, 52) OVER w_momentum AS sales_last_year,
                
                -- 4. Rolling Momentum (Shifted 1 to prevent leakage)
                AVG(LN(weekly_sales + 1)) OVER (
                    PARTITION BY store, dept 
                    ORDER BY date 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) AS rolling_4_wk_log_sales_avg,
                
                -- 4b. Rolling Momentum (Raw Dollars for Operations)
                AVG(weekly_sales) OVER (
                    PARTITION BY store, dept 
                    ORDER BY date 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) AS rolling_4_wk_sales_avg,
                
                -- 5. Macroeconomic CPI lag (3 months ~ 12 weeks)
                LAG(cpi, 12) OVER (
                    PARTITION BY store 
                    ORDER BY date
                ) AS cpi_lag_3_month

            FROM merged_data
            WINDOW w_momentum AS (PARTITION BY store, dept ORDER BY date)
            ORDER BY date, store, dept
        """

        con.execute(gold_sql)

        logger.info('Extracting into Polars for final storage optimization...', layer='gold')
        lf = pl.from_arrow(con.execute("SELECT * FROM gold_master").arrow())
        
        # Optimize Storage Types
        import polars.selectors as cs

        
        # 1. Cast Integers to Floats: ML frameworks (MLflow, BentoML) crash if they encounter 
        # missing values inside integer columns during inference. Floats handle NaNs natively.
        lf = lf.with_columns(cs.integer().cast(pl.Float64))
        
        # 2. Dictionary Encoding (Categoricals): Saves ~70% disk space and RAM.
        for col in cfg.get("data.features.categorical"):
            lf = lf.with_columns(pl.col(col).cast(pl.String).cast(pl.Categorical))
        
        gold_path = Path(cfg.get("data.paths.gold_data"))
        logger.info('Writing heavily optimized Parquet', path=str(gold_path))
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        lf.write_parquet(str(gold_path))
     
        logger.info('Closing DuckDB connection', layer='gold')
        con.close()

        duration = round(time.time() - start_time, 2)
        logger.info('Gold layer transformation completed', layer='gold', status='success', duration_seconds=duration)

    except Exception as e:
        logger.error('Gold layer transformation failed', layer='gold', error=str(e))
        raise

if __name__ == "__main__":
    create_gold_layer()