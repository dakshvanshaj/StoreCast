# High-Performance Medallion Data Architecture

**StoreCast** operates on a decoupled, hardware-constrained local environment mimicking a multi-million-dollar enterprise scalable cloud ecosystem. To achieve this, the data architecture relies on a highly efficient **"Tri-Stack" Medallion ETL Pipeline** using purely open-source tooling, strictly orchestrated via **Data Version Control (DVC)**.

![alt text](images/medallian.svg)

## Architecture Diagram & DVC Orchestration

Because data volumes scale beyond standard `git` capabilities, we rely on DVC to track massive Delta and Parquet files remotely (via DagsHub) while locally orchestrating the pipeline graph. 

One of the most powerful features of our pipeline is **DVC Caching (`dvc repro`)**. DVC algorithmically hashes the inputs and python scripts of every stage. If our PySpark ingestion logic and Bronze data haven't changed, but we adjust our Gold DuckDB query, `dvc repro` will instantly skip the Bronze and Silver extraction stages (pulling directly from cache) and only execute the Gold stage—saving massive compute resources!

## The Single "Data Lakehouse" Ecosystem
Traditionally, data copied from a messy Data Lake into an expensive, separate Data Warehouse. StoreCast abandons this by leveraging a single **Data Lakehouse**. The entire system exists on local files, logically separated into three maturity layers:

- **Bronze (Raw Ingestion):** The immutable historical record. PySpark faithfully dumps the external CSVs into robust `Delta` folders but makes no attempt to clean them. 
- **Silver (Cleansed Source of Truth):** The enterprise anchor layer. Polars applies strict schema enforcement, clipping, and anomaly resolution. Because it is a `Delta` table, it retains full ACID logic and Time Travel if data corruption occurs.
- **Gold (Aggregated / OBT):** We intentionally abandon both `Delta` format and complex `Star Schemas` here. Parquet does not have Time Travel, and we do not use Fact/Dimension tables. Instead, DuckDB flattens everything into a highly aggregated **One Big Table (OBT)** (`master_sales.parquet`). This simple Parquet file is optimized strictly for maximal read-speed, designed to be ingested flawlessly by our Machine Learning models and BI Dashboards without requiring them to join anything.

## The "Tri-Stack" Tooling Rationale

Rather than defaulting to `pandas` or adopting heavyweight infrastructure like `dbt`, StoreCast uses a highly specialized "Tri-Stack" approach to data processing. 

The core philosophy is **Right-Sizing Compute**: 
*   **Horizontal Scaling** is for **Distributed I/O**: Best for reading 10,000 messy external files simultaneously.
*   **Vertical Scaling** is for **Logic & Math**: Best for processing that data once it is structured and compressed. 

### 1. PySpark (Bronze Ingestion: Horizontal Scaling)
- **The Phase:** Raw data extraction and distributed landing.
- **The Justification:** PySpark handles the "Heavy Lift" of Distributed Ingestion—streaming raw retail CSVs, inferring massive schemas, and writing them out as robust `Delta Lake` tables. By using horizontal scaling here, we can ingest Petabyte-scale data without rewriting code. 

### 2. Polars (Silver Transformation: Vertical Scaling)
- **The Phase:** Cleaning anomalies, deduplicating, and standardizing datatypes.
- **The Justification:** Polars is a blazingly fast **Vertical Scaling** engine. Silver transformations are CPU-bound and benefit from **Local Memory Efficiency**. By using a single beefy node instead of a cluster, we eliminate the **"Shuffle Tax"** (moving data between nodes) and the **"Startup Latency"** (waiting 5-7 minutes for a cluster to provision). Polars often finishes the job before a Spark cluster has even finished booting up.

### 3. DuckDB (Gold Modeling & Feature Engineering)
- **The Phase:** Analytics flattening, deep time-series window functions.
- **The Justification:** DuckDB executes native vector-based **out-of-core SQL** directly against our Silver Delta tables. It maximizes vertical efficiency by avoiding the compute-heavy network shuffles and "Master Node" overhead of distributed SQL engines. This provides the power of a Data Warehouse with zero "Always-On" infrastructure costs.

## Cloud Scalability Parity
While this current architecture is engineered to execute locally with a **decoupled development footprint**, it is designed for perfect 1:1 cloud production parity:
- **PySpark:** Natively deploys to AWS EMR or Databricks for massive horizontal scaling if the data grows into Petabytes.
- **Polars & DuckDB:** Easily containerized into Docker, running seamlessly inside cheap AWS Fargate instances or Kubernetes pods, executing the exact same decoupled SQL against an S3 bucket without requiring any code refactoring.

## The Quality Gates (Great Expectations)
A Medallion Lakehouse is useless if the data is poisoned. 
Between the Silver and Gold layers, we enforce strict **Data Contracts** mapped by Great Expectations (GX).

- **Silver Gates:** Ensure primary keys (`Store`, `Dept`, `Date`) are mathematically unique (no cartesian explosion duplicates), sales constraints (no negative numbers) are held and other business rules are followed.
- **Gold Gates:** Ensure our advanced DuckDB features successfully mapped without returning catastrophic `NULL` explosions (except for expected mathematical warm-up periods) and other business rules are followed.
- **Automated Docs:** Each run automatically writes HTML Data Docs detailing the exact mathematical compliance of the pipeline. If a test fails, the pipeline errors out before the ML script can poison its weights!

## Pipeline Idempotency & Determinism
One of the most critical requirements of a production pipeline is **Idempotency**—the ability to run the pipeline multiple times without causing data corruption or duplication. 

- **Data Idempotency:** If the pipeline fails midway or is run twice by mistake, our Silver Polars logic utilizes strict primary-key upserts and distinct group-bys to ensure no cartesian duplicates ever propagate downstream.
- **Environment Determinism:** We abandoned legacy `pip` and `conda` in favor of **`uv`**. By using `uv` with a synchronized `pyproject.toml` and `uv.lock`, we mathematically guarantee that the python memory space running the DuckDB SQL window functions locally is identical to the containerized environment running in the cloud.

## Data Serialization & Storage Formats
The physical files that DVC pushes to our remote storage are strictly enforced big-data formats, not standard CSVs.

### Delta Lake (Bronze & Silver)
Our Bronze layer (PySpark) and Silver layer (Polars) persist data exclusively as **Delta Tables**. 
- **What is it?** A Delta table is not a single file; it is a directory containing highly-compressed `.parquet` files wrapped in a strict JSON transaction ledger called the `_delta_log`. 
- **ACID Guarantees:** In standard data engineering, if a pipeline fails halfway through writing a 2GB CSV file, the file is corrupted. With Delta, the ingestion is "All or Nothing." If a job fails, the JSON transaction log aborts the commit, and the table remains perfectly intact, shielding downstream consumers from poisoned data.
- **Data Time Travel:** Because Delta Lake utilizes "append-only" mechanics (it never overwrites old Parquet files, it just writes new ones and updates the ledger pointer), the history is perfectly preserved. This enables **Time Travel**. If an Engineer accidentally deletes critical rows, we can simply query the table locally or in DuckDB using `SELECT * FROM silver_table VERSION AS OF 2` or `TIMESTAMP AS OF '2026-01-01'` to instantly read the data exactly as it existed on that date. 

### Parquet (Gold Layer & BI Consumption)
The finalized Gold layer is written as strictly-typed **Parquet** files.
- **Columnar Storage:** Parquet stores data column-by-column rather than row-by-row. If a dashboard needs the sum of `weekly_sales`, the engine does not have to load the other 20 columns (temperatures, fuel prices, etc.) into RAM. It reads only the compressed sales column, resulting in 100x performance gains over CSVs.
- **Decoupled Remote ELT:** In development, dashboards query the local Parquet files. However, in production, engines like DuckDB use HTTP Range Requests to query the remote DagsHub Parquet files directly. DuckDB reads the Parquet metadata footer, identifies exactly where the necessary data lives on the remote server, and downloads only those specific chunks over the network. This provides blazing fast, decoupled cloud ELT!
