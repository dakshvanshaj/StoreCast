# Data Architecture & Technical Tradeoffs

In a standard machine learning portfolio project, it is common to load a CSV into Pandas, train a model, and call it a day. In building enterprise architecture, this approach fails immediately under load. 

StoreCast implements a hybrid **Medallion Architecture**, mixing distributed computing (PySpark) with lightning-fast single-node engines (Polars, DuckDB) to simulate a modern, cost-aware Databricks Lakehouse. This document explicitly outlines *why* we decoupled our ingestion and processing layers.

## The Big Question: Why use PySpark at all if Polars is so fast?
A common architectural question arises: *If Polars is going to read the data into memory anyway for the Silver layer, why do we bother using PySpark to extract the CSVs in the Bronze layer? Couldn't Polars just read the raw CSVs directly?*

If our retail data is 15 Megabytes, yes. If our retail data is 15 Terabytes, **absolutely not.**

In the real world, the Bronze layer deals with unbounded, massive raw data dumps from thousands of Point-of-Sale registers. 
1. **The RAM Limit:** Polars and Pandas are fundamentally in-memory (or out-of-core chunked) engines. They cannot effectively ingest a monolithic 15-Terabyte CSV file without immense infrastructure overhead.
2. **Distributed IO:** PySpark does not load the entire file into a single machine's memory. It distributes the reading of the raw CSV across 100+ worker nodes, parses the messy strings, compresses them natively, and writes them out physically to disk as partitioned Delta files. PySpark's sole job is to safely extract the data from the external world and land it in our Lakehouse.

## Why switch to Polars for Silver Transformation?
Once PySpark has landed the data as highly compressed, partitioned Delta/Parquet files in our Lakehouse, the engineering bottleneck shifts. We are no longer processing 15 Terabytes of raw text; we are executing complex data cleaning workflows (forward-filling time-series values, window functions, and joining).

1. **Columnar Compression:** The raw 15TB CSV is just uncompressed text. When PySpark writes it to Delta/Parquet, it uses heavy columnar compression (like Snappy or Zstd). The physical size on disk drops by 80-90% immediately.
2. **Predicate Pushdown & Partitioning:** Because PySpark already cleanly partitioned the Bronze data on disk, Polars does not have to read the entire dataset into memory. Using Delta/Parquet reading logic, Polars scans the metadata and only pulls the exact folders/bytes required into RAM.
3. **Out-of-Core Streaming Mode:** What if the compressed, partitioned data is *still* too big for our Polars server's RAM? Polars has a Lazy Streaming Pipeline (`pl.scan_parquet()`) that processes the data in tiny batches, passing it through the CPU without ever holding the full dataset in memory at once.
4. **Right-Sizing Compute (Financial Cost):** Because Polars can stream heavily compressed data so efficiently, launching a massive PySpark cluster to execute a simple `.clip()` or `.fillna()` on structured data is financially irresponsible. A distributed Spark cluster costs hundreds of dollars an hour. Polars can execute those exact same row transformations locally using vectorized Rust, easily fitting inside a cheap $10/month cloud container.
3. **The Data Scientist Experience:** Data Scientists often struggle fighting PySpark's lazy-evaluation JVM errors for feature engineering. Providing them with lightning-fast Python/Rust APIs (Polars) drastically decreases Time-to-Market for generating Machine Learning Models.

## Algorithm Selection Tradeoffs: Market Basket Analysis
In standard Market Basket Analysis, algorithms like **Apriori** or **FP-Growth** are the undisputed gold standards. They are designed to find association rules (e.g., "If a customer buys Milk, they also buy Bread"). So why don't we use them in StoreCast?

This is a classic example of **Data Dimensionality Constraints** shaping model architecture:
1. **The Transaction-Level Requirement:** Apriori and FP-Growth mathematically require *transaction-level* data. The dataset must look exactly like an itemized cash-register receipt (`[Transaction_101: Milk, Bread, Eggs]`).
2. **Our Aggregated Reality:** Our dataset's lowest grain is `Store + Department + Week`. We do *not* have individual customer receipts; we only have the total gross sales for an entire department aggregated over a 7-day period. It is mathematically impossible to feed "Weekly Sales = $10,000" into an Apriori algorithm.

**The Solution (Pearson Correlation Proxy):**
When denied ideal POS (Point of Sale) data, we must engineer a proxy. To determine if retail items are "bought together", we check their structural movement across time: *If Department 90's gross sales perfectly spike and dip alongside Department 87's gross sales over 143 consecutive weeks, they are fundamentally linked in the macro consumer behavior cycle.* 

Using **Pearson Correlation Matrixing** on time-series aggregated data is the industry-standard architectural hack to extract structural association insights when raw transaction-level data is unavailable or aggressively compressed.

## Why DVC over dbt for Pipeline Orchestration?
In modern data engineering, **dbt (Data Build Tool)** is the undisputed industry standard for the transformation layer. It excels at turning raw data inside a cloud Data Warehouse (like Snowflake or BigQuery) into clean Silver/Gold tables using templated SQL, while automatically managing the execution DAG (dependency graph).

So why didn't we use it in StoreCast? 

1. **The Infrastructure Constraint:** dbt requires an underlying database engine to run its SQL against (usually a massive, expensive cloud warehouse). By choosing DuckDB and Polars on local Delta Lake files, we maintain our "decoupled, lightning-fast local compute" constraint. Standing up a warehouse just to use dbt would be massive overkill.
2. **Machine Learning vs. Business Intelligence:** dbt is built for *Analytics Engineers* feeding Business Intelligence dashboards. However, our persona is a *Machine Learning Engineer*. We need to execute complex mathematics (like training Anomaly Detection isolation forests or target encoding) that SQL handles poorly. A pure Python ecosystem gives us the flexibility to seamlessly mix data engineering with advanced Scikit-Learn logic.
3. **The DVC Solution:** We replicated dbt's primary superpower—the DAG orchestration—using **DVC (Data Version Control)**. Instead of dbt figuring out our SQL dependencies, DVC dynamically tracks our Python scripts (`ingest_bronze.py` → `clean_silver.py` → `create_gold.py`), caches the intermediate outputs, and gives us full data-versioning rigor all without leaving Python.

## Why We Intentionally Skipped a "Feature Store" (Feast / Hopsworks)

**What is a Feature Store?**
A Feature Store (like Feast or Hopsworks) is a centralized application used by large enterprises. Think of it like a massive pre-chopped ingredient fridge in a commercial kitchen. If Uber has 50 different Data Science teams building 100 different machine learning models, 40 of those teams might need a feature called `driver_average_rating`. Without a Feature Store, all 40 teams write their own separate SQL pipelines to calculate it, wasting millions of dollars in redundant compute. A feature store centralizes this: Data Engineers calculate it *once*, save it to the store, and all 50 teams just `import` it. Feature stores also hold those pre-calculated numbers in ultra-fast memory (like Redis) so models making live predictions in milliseconds (like credit card fraud) can grab the data instantly.

**Why did StoreCast reject it?**
- **Infrastructure Decoupling Strategy:** Deploying scalable Feature Stores (like Feast) requires maintaining dedicated "Always-On" cloud infrastructure (like Redis clusters). Our architecture intentionally rejects this lock-in, favoring decentralized, "Zero-Copy" compute that drastically minimizes cloud overhead.
- **Batch SLA:** StoreCast performs *Weekly Batch Forecasting*. Our demand planners do not need predictions resolved in 50 milliseconds; they view a PowerBI dashboard updated once a day. 
- **The "Gold Layer" as an Implicit Store:** We only have one primary pipeline. When DuckDB finishes engineering the `52-week lag` and `4-week moving average` features, it saves them directly into `master_sales.parquet`. We version that exact file with DVC. For our specific use case, that finalized Parquet file acts as our "Batch Feature Store" perfectly well without any of the enterprise overhead.

## Summary of the Tri-Stack Philosophy:
- **PySpark (Bronze):** Maximizes horizontal distributed scale for raw unbounded data extraction.
- **Polars/DuckDB (Silver/Gold):** Maximizes vertical speed and minimizes cloud computing costs for analytical processing and feature engineering on structured, partitioned data.
- **DVC (Orchestration):** Replaces dbt by providing granular dependency tracking and data versioning entirely within a Python-native ML environment.
