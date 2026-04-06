# Gold Layer Pipeline (`src/data/create_gold.py`)

**Owner:** Data Engineering / Data Science Team
**Engine:** DuckDB

## Purpose
This script is responsible for transitioning the verified, clean data from the Silver Lakehouse into a high-performance Analytical Master Table (Gold Layer) stored in Parquet format. This artifact serves as the single source of truth for both Business Intelligence Dashboards and Machine Learning models.

## Architectural Trade-offs & Design Decisions

### 1. Why is this script so strictly declarative (SQL) instead of Pythonic?
In legacy ETL paradigms, feature engineering scripts are notoriously bloated because they intermix data cleansing, data type enforcement, and imputation alongside complex aggregations.
Because we strictly adhere to the **Medallion Architecture**, all data sanitization was handled upstream in `clean_silver.py`. The Gold layer is reserved entirely for **Dimensional Modeling** and **Analytical Aggregations**. 

### 2. Why DuckDB?
We use DuckDB to execute a massive `JOIN` across our `sales`, `features`, and `stores` Silver Delta tables. DuckDB contains a modernized, vectorized SQL engine capable of processing window functions (like rolling averages and lagged seasonality) significantly faster and with less memory overhead than Pandas `.merge()` and `.shift()` operations. By using `delta_scan()`, DuckDB queries the delta lake directly with zero-copy.

### 3. Why export to Parquet instead of Delta?
While Delta Lake guarantees ACID transactions and time-travel for our data lake, the Gold layer output is consumed directly by Machine Learning libraries (like Scikit-learn and XGBoost) which natively expect flattened Parquet files. Since the Gold layer is a read-only, fully rebuilt artifact, the overhead of a Delta log is unnecessary.

## Engineered Features
The script leverages DuckDB's advanced Window Functions to engineer the insights discovered during our Silver EDA:
- **Temporal Features:** `month` and `week_of_year` extracted directly from the Date.
- **Seasonality Lags:** `sales_last_year` computed using a 52-week lag partitioned by store/department.
- **Velocity Features:** `rolling_4_wk_sales_avg` using a bounded preceding window.
- **Macro Lags:** `cpi_lag_3_month` (12-week lag) accounting for reporting delays in real-world macro-economic data.
