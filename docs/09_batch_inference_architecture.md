# Phase 5: The Batch Inference Engine

The ultimate goal of ML engineering is not to generate high R-Squared metrics—it is to deliver actionable data to the business. 

While `trainer.py` and `optimizer.py` train and test the model on *historical* data, `batch_inference.py` is the operational engine. It is the script that will run every Sunday night to answer the question: *"How many units will we sell next week?"*

## What Does The Script Actually Do?

Here is the step-by-step breakdown of how the Inference Engine operationalizes the ML pipeline:

### 1. Dynamic Model Retrieval
The script completely ignores local `.pkl` files. It interfaces directly with the DagsHub Cloud Registry and calls `models:/StoreCast_XGBoost@production`. 
By pulling the model dynamically via alias, the engineering team can run `optimizer.py`, train a better model, and tag it as `@production`. The next day, the Batch Inference Engine will automatically fetch the new champion model without requiring a single code change.

### 2. Live Data Simulation
In a real-time production system, fresh data arrives via data pipelines (e.g., Kafka or Snowflake). For the StoreCast pilot, we simulate this "unseen future data" by reaching into DuckDB's Gold analytical layer (`gold_sales.parquet`) and isolating the final 1,000 rows (`lf.tail(1000)`). These 1,000 rows act as our mathematically engineered feature matrix for the impending forecast window.

### 3. Strict Schema Coercion
XGBoost and MLflow Pipeline deployments are notoriously brittle if the incoming data types shift. If training data passed a `Float64`, but inference data passes an `Int32`, the model crashes. The script uses Polars' rigorous `.cast()` typing to mathematically guarantee the schema perfectly matches the training environment before making a single prediction.

### 4. High-Throughput Scoring
The engine feeds the 1,000 rows into the pipeline and calculates `pipeline.predict()`. The model takes into account the `lag_52`, `IsHoliday`, and `CPI` for each row, collapsing them into single numerical dollar estimates (the predicted revenue).

### 5. Contextual Output Generation
Raw arrays like `[15000, 20000, 18000]` mean absolutely nothing to a Demand Planner. The engine intelligently maps the predictions back onto the original `df_live` dataframe and extracts only `['store', 'dept', 'date', 'weekly_sales', 'Predicted_Weekly_Sales']`. 

It cleanly exports this table to `data/04_predictions/batch_forecast_results.csv`, establishing a frictionless bridge for PowerBI or Tableau dashboards to ingest the data instantly.
