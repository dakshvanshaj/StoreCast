# Data Architecture & Technical Tradeoffs

In a standard machine learning portfolio project, it is common to load a CSV into Pandas, train a model, and call it a day. In building enterprise architecture, this approach fails immediately under load. 

StoreCast implements a hybrid **Medallion Architecture**, mixing distributed computing (PySpark) with lightning-fast single-node engines (Polars, DuckDB) to simulate a modern, cost-aware Databricks Lakehouse. This document explicitly outlines *why* we decoupled our ingestion and processing layers.

## The Big Question: Why use PySpark at all if Polars is so fast?
A common architectural question arises: *If Polars is going to read the data into memory anyway for the Silver layer, why do we bother using PySpark to extract the CSVs in the Bronze layer? Couldn't Polars just read the raw CSVs directly?*

If our retail data is 15 Megabytes, yes. If our retail data is 15 Terabytes, **absolutely not.**

In the real world, the Bronze layer deals with unbounded, massive raw data dumps from thousands of Point-of-Sale registers. 
1. **Distributed Extraction (Horizontal Scaling):** PySpark does not load the entire file into a single machine's memory. It distributes the reading of the raw CSV across 100+ worker nodes, parses the messy strings, compresses them natively, and writes them out physically to disk as partitioned Delta files. This **Horizontal Scaling** allows us to ingest Petabyte-scale data that would instantly crash a single machine.
2. **The "Heavy Lift" Component:** PySpark's sole job is to safely extract the data from the unbounded external world and land it in our Lakehouse. It acts as our "Distributed Ingestion Engine."

## Why switch to Polars for Silver Transformation?
Once PySpark has landed the data as highly compressed, partitioned Delta/Parquet files in our Lakehouse, the engineering bottleneck shifts. We are no longer processing 15 Terabytes of raw text; we are executing complex data cleaning workflows (forward-filling time-series values, window functions, and joining).

1. **Vertical Efficiency (The Billion Row Challenge):** Polars is a high-performance **Vertical Scaling** engine. Because it is written in Rust and utilizes multithreading and SIMD instructions, it can process hundreds of millions of rows on a single beefy machine faster than Spark can shuffle them across a network.
2. **Columnar Compression:** The raw 15TB CSV is just uncompressed text. When PySpark writes it to Delta/Parquet, it uses heavy columnar compression (like Snappy or Zstd). The physical size on disk drops by 80-90% immediately, making the data manageable for vertical engines.
3. **Predicate Pushdown & Partitioning:** Because PySpark already cleanly partitioned the Bronze data on disk, Polars does not have to read the entire dataset into memory. Using Delta/Parquet reading logic, Polars scans the metadata and only pulls the exact folders/bytes required into RAM.
4. **Out-of-Core Streaming Mode:** What if the compressed, partitioned data is *still* too big for our Polars server's RAM? Polars has a Lazy Streaming Pipeline (`pl.scan_parquet()`) that processes the data in tiny batches, passing it through the CPU without ever holding the full dataset in memory at once.
5. **The "Efficiency Gap" & Right-Sizing Compute (Financial Cost):** Scaling **Horizontally** (adding more Spark nodes) is exponentially more expensive than scaling **Vertically** (adding RAM/CPU to one machine). 
    *   **The Shuffle Tax:** When a Spark cluster performs a transformation, it must move data across the network between nodes (the "Shuffle"). This is extremely expensive in terms of time, I/O, and cloud network costs. On one beefy Polars node, there is **zero network shuffle**; all the data lives in the same physical RAM sticks. 
    *   **Startup Latency:** Provisioning a 20-node Spark cluster (Databricks/AWS EMR) often takes **3-7 minutes** just to prepare the virtual machines. For many transformation scripts, Polars has finished the job before the Spark cluster has even finished saying "Hello."
    *   **The Master Node Overhead:** A Spark cluster requires a "Master Node" (the brain) that manages workers but doesn't actually process any data. On a single node, 100% of your compute budget goes directly to processing your retail sales.
6. **The Swappable Layer (Modular Design):** Because our compute is decoupled from the storage, this is a "No-Regret" architecture. If our data volume eventually exceeds what a single machine can handle even with vertical scaling, we simply swap the Polars transformation script for a Spark script. The Delta storage foundation and Gold SQL remain 100% identical.
7. **The Data Scientist Experience:** Data Scientists often struggle fighting PySpark's lazy-evaluation JVM errors for feature engineering. Providing them with lightning-fast Python/Rust APIs (Polars) drastically decreases Time-to-Market for generating Machine Learning Models. Debugging a single-node script is significantly easier than digging through 20 different worker logs.

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

## Phase 4 & Phase 5 ML Pipeline Architectural Tradeoffs

In Phase 4 (Experimentation) and Phase 5 (Production Deployment), we make several highly intentional architectural choices regarding our ML framework to balance development speed with production efficiency.

### 1. The TransformedTargetRegressor Hack vs Native SQL
**The Problem:** Our target variable (`weekly_sales`) requires a log transformation (to handle zero bounds and fat-tails). If you train a model on Log Sales, your predictions will be in logs. In production, business stakeholders need predictions in real dollars.

**The Naive Solution:** Calculate the log in DuckDB `log1p()`, train a native model, and then try to evaluate it. 
**Why we rejected it initially:** In our experimentation phase, we use `Optuna` and `cross_val_score` to find hyperparameter combinations that minimize our Business Metric (Weighted MAPE on real dollars). If we pass a naked model into an Optuna cross-validation loop, Optuna only sees the log-errors, not the dollar-errors. Writing manual loops to extract, `.expm1()`, and calculate WMAPE for 5 folds across 100 trials is an unmaintainable nightmare.

**The StoreCast Solution (Phase 4):**
We temporarily wrap our XGBoost algorithm inside Scikit-learn's `TransformedTargetRegressor`. This allows us to pass real dollars `y_train` into `.fit()`. Under the hood, sklearn takes the log automatically, trains the nested XGBoost, and automatically applies `np.expm1()` when predicting. This abstraction allows Optuna to elegantly score real dollar WMAPE.

### 2. Ditching Scikit-Learn in Production (Phase 5)
Once Optuna finds our optimal hyperparameters, we explicitly **drop Scikit-learn** and train a native `xgboost.Booster` for the Bentoml/FastAPI serving layer.

**Why? (The Interview Answer):**
- **Latency & Overhead:** Scikit-Learn object wrappers carry metadata overhead that slows down milliseconds-critical inference. 
- **Serialization Vulnerabilities:** Loading sklearn `.pkl` files opens massive security and versioning vulnerabilities. A native booster can be saved cleanly as JSON or C++ binaries.
- **Microservice Design:** By calculating the target transformation (`log1p`) directly in our DuckDB gold script, and applying the inverse transformation (`expm1`) natively in our Python API serving endpoint, we maintain a pure, un-obfuscated inference microservice.

### 3. The Re-Integration of Feast (Feature Store)
**What is a Feature Store?**
Think of it like a massive pre-chopped ingredient fridge. If multiple DS teams need a specific feature (like `lag_4_sales`), a feature store calculates it *once* and serves it to both offline training and online low-latency inference endpoints (often via Redis).

**Why we initially rejected it:** We had $0 budget and our batch SLA (Weekly Forecasts) meant a flat DuckDB `master_sales.parquet` file perfectly acted as our implicit "offline" feature store.

**Why we incorporate it in Phase 5:**
To mature our portfolio into a **2026 Enterprise-Grade Platform** (like Uber Michelangelo or DoorDash), we implement Feast alongside BentoML. 
- **The Decoupling Tradeoff:** Instead of forcing `RobustScaler` and `SimpleImputer` inside our Scikit-Learn pipelines (which causes massive data-leakage threats if scaled dynamically across a server fleet), we use Feast to retrieve identical, point-in-time correct raw feature inputs for both our Jupyter training notebooks and our BentoML online serving cache. Then, the natively robust Gradient Boosters process them flawlessly.

## Data Modeling: Star Schema vs. One Big Table (OBT)

Historically, generating a **Star Schema** (normalizing data into a central `fact_sales` table surrounded by smaller dimension tables like `dim_store` and `dim_date`) was the undisputed gold standard. If you repeatedly wrote the text `"Walmart Store 45, Texas, 100k sqft"` millions of times in a flat file, you wasted millions of dollars in expensive 1990s hard-drive storage. Normalizing it into a tiny `dim_store` table saved space. 

**Why did StoreCast intentionally abandon the Star Schema in the Gold Layer?**

We chose to completely denormalize our Gold Parquet layer into **One Big Table (OBT)** for three architectural reasons:

1. **Machine Learning Constraints:** Algorithms like Random Forest and XGBoost expect a single, flattened 2D matrix (`X_train`, `y_train`). If you feed a Data Scientist a Star Schema, the very first thing they must do is write computationally expensive SQL `JOIN`s to flatten it back out before they can call `model.fit()`. Our Gold layer pre-flattens this natively so the data is instantly ML-ready.
2. **Compute is more expensive than Storage:** Today, AWS S3 storage is virtually free, but CPU clusters are expensive. If you build a Star Schema, every time your PowerBI Dashboard refreshes, it must burn CPU cycles to dynamically `JOIN` massive tables together. By pre-joining everything into an OBT via DuckDB in the pipeline phase, downstream dashboards do zero runtime joining—they just read straight down the columns blisteringly fast.
3. **Parquet Dictionary Encoding:** Modern Parquet uses advanced dictionary compression. Even if we write `"Store 45"` five million times in an OBT, Parquet automatically realizes it is a repeating string, assigns it a tiny binary code underneath, and compresses it down to practically zero bytes. The ancestral "Star Schemas save storage space" argument is completely negated by modern column-oriented encodings.

## Summary of the Tri-Stack Philosophy:
- **PySpark (Bronze):** Maximizes horizontal distributed scale for raw unbounded data extraction.
- **Polars/DuckDB (Silver/Gold):** Maximizes vertical speed and minimizes cloud computing costs for analytical processing and feature engineering on structured, partitioned data.
- **DVC (Orchestration):** Replaces dbt by providing granular dependency tracking and data versioning entirely within a Python-native ML environment.

## Time-Series Cross Validation Tradeoffs

When transitioning into Phase 4 Hyperparameter tuning (Optuna), evaluating models using standard K-Fold Cross Validation introduces a fatal architecture flaw called **Future Data Leakage**. If Fold 1 randomly samples November 2012 to predict November 2011, the model learns future macro-economic trends that will not exist in production, artificially inflating evaluation metrics.

**The Theoretical Fix (Walk-Forward Validation):**
The standard industry fix is Scikit-Learn's `TimeSeriesSplit`, which uses expanding windows (Train on Jan-Mar to validate Apr, then Train on Jan-Apr to validate May). While mathematically rigorous, it is computationally bruising (requiring 3+ complete model retrains for *every single* Optuna trial) and heavily complicates custom scorer injection (like our Holiday-weighted WMAPE metric).

**The StoreCast Solution (The 3-Way Chronological Split):**
To adhere to our "Keep it Lean and Simple" project philosophy while preserving airtight data hygiene, we bypassed K-Fold entirely in favor of a strict 3-way chronological slice:
1. **Train Set (First 70%):** Provided to the algorithm for coefficient generation.
2. **Validation Set (Next 15%):** The "Immediate Future" used strictly by Optuna's Bayesian search to calculate WMAPE and tune hyperparameters independently.
3. **Test Set (Final 15%):** Locked in memory and deliberately hidden from Optuna. Used exactly once at the end of the pipeline to objectively evaluate the mathematically "unseen" future.

While not as statistically bulletproof as 10-fold Walk-Forward Validation, this proxy delivers 95% of the data-hygiene benefits at a fraction of the computational latency and engineering overhead, a paramount consideration for lean pipeline scaling.
