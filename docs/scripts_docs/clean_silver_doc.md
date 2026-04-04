# Silver Transformation Pipeline (`clean_silver.py`)

**Path:** `src/data/clean_silver.py`  
**Execution:** `uv run python -m src.data.clean_silver`  
**Layer:** Bronze ➔ Silver  
**Engine:** Polars 1.0+

## Overview
This script represents the second stage of our Medallion Architecture. It reads the raw, distributed Delta Lake partitions ingested during the Bronze phase and performs highly targeted, in-memory data cleaning using **Polars**. 

By utilizing Polars (a Rust-based vectorized engine) alongside Delta Lake's native reading integration, this script executes complex time-series row-level transformations on 450,000+ records in under 1 second without requiring an expensive distributed JVM cluster.

## Business Logic & Transformation Rules
The logic in this script is entirely driven by the insights discovered during the `02_bronze_eda.ipynb` Data Analyst phase. We identified three major pipeline-breaking anomalies and codified their resolutions:

1. **Clipping Negative Returns (`sales_df`)**
   - **Problem:** `weekly_sales` contained negative values representing customer merchandise returns. Training a forecasting model on negative demand mathematically corrupts the objective function.
   - **Resolution:** Instead of dropping the rows (which breaks chronological continuity for time-series forecasting), we applied `.clip(lower_bound=0.0)`. This neutralizes the anomaly without destroying the temporal sequence.

2. **Strict-Casting & Filling MarkDowns (`features_df`)**
   - **Problem:** Historical promotional data (2010-2011) was unrecorded, containing literal `"NA"` strings and occasional negative data-entry errors.
   - **Resolution:** Polars operates purely in strict Rust logic and will crash if it encounters strings during explicit numeric casts. We leveraged `.cast(strict=False)` to force these uncoercible text elements into native nulls, then chained `.fill_null(0.0)` and `.clip(0.0)` to normalize all missing markdown data directly into a $0 discounted base value.

3. **Macro-Economic Forward Filling (`features_df`)**
   - **Problem:** `CPI` and `Unemployment` data lag behind retail reporting, leaving nulls at the tail-end of our time series.
   - **Resolution:** We cannot mathematically fill macroeconomic indicators with `0`. We applied a `.fill_null(strategy="forward")` rolling window function to carry the last known government metrics forward.

4. **Date Parsing**
   - Fixed explicit European date syntaxes (`%d/%m/%Y`) into highly optimized, native Date types preparing the engine for high-speed DuckDB Star-Schema merges.

## Idempotency
Just like the Bronze ingestion script, this is a fully **idempotent** operation (`mode="overwrite"`). 
If run multiple times, it will safely recreate the `data/silver` Delta tables without generating duplicates. Because Delta Lake supports time-travel, previous versions of the Silver tables are securely retained internally via the `_delta_log` metadata for transaction durability.

## Framework Constraints
- **Structured Logging:** Uses `structlog` to natively emit JSON-parsable Key/Value logs for Grafana Loki log aggregation.
- **Fail-Safe Processing:** Wrapped entirely in a `try/except` block to ensure execution environments gracefully catch and trace any upstream memory or syntactical panics.
