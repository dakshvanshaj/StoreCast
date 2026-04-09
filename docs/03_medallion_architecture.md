# High-Performance Medallion Data Architecture

**StoreCast** operates on a decoupled, hardware-constrained local environment mimicking a multi-million-dollar enterprise scalable cloud ecosystem. To achieve this, the data architecture relies on a highly efficient **"Tri-Stack" Medallion ETL Pipeline** using purely open-source tooling, strictly orchestrated via **Data Version Control (DVC)**.

## Architecture Diagram & DVC Orchestration

Because data volumes scale beyond standard `git` capabilities, we rely on DVC to track massive Delta and Parquet files remotely (via DagsHub) while locally orchestrating the pipeline graph. 

One of the most powerful features of our pipeline is **DVC Caching (`dvc repro`)**. DVC algorithmically hashes the inputs and python scripts of every stage. If our PySpark ingestion logic and Bronze data haven't changed, but we adjust our Gold DuckDB query, `dvc repro` will instantly skip the Bronze and Silver extraction stages (pulling directly from cache) and only execute the Gold stage—saving massive compute resources!

```mermaid
flowchart LR
    %% Styling Settings
    classDef dvc_stage fill:#0d1117,stroke:#3b82f6,stroke-width:2px,color:#fff;
    classDef storage fill:#075985,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef compute fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#f8fafc;
    classDef qa fill:#581c87,stroke:#c084fc,stroke-width:2px,color:#f8fafc;
    classDef remote fill:#9a3412,stroke:#fdba74,stroke-width:2px,color:#f0fdf4;
    classDef tool fill:#1e293b,stroke:#94a3b8,stroke-width:1px,color:#38bdf8,stroke-dasharray: 4 4;
    classDef memory fill:#b45309,stroke:#f59e0b,stroke-width:2px,color:#fff;
    classDef delta_feature fill:#064e3b,stroke:#10b981,stroke-width:1px,color:#fff,stroke-dasharray: 3 3;

    subgraph Pipeline [DVC Orchestration Pipeline]
        style Pipeline fill:#0f172a,stroke:#334155,stroke-width:2px,color:#e2e8f0;
    
        %% Bronze
        A[Raw CSVs] -->|Distributed Extract| B[(Bronze: Delta Tables)]
        T1(PySpark Engine) -.-> B
        class A,B storage
        class T1 tool
        
        %% Silver
        B -->|Stream to RAM| Mem1((System RAM))
        Mem1 -->|Out-ofCore Write| C[(Silver: Delta Tables)]
        T2(Polars Lazy Engine) -.-> Mem1
        C --> D[Silver Data Contract]
        class C storage
        class Mem1 memory
        class D qa
        class T2 tool
        
        %% Gold
        D -->|Validation Pass| Mem2((System RAM))
        C -->|Stream to RAM| Mem2
        Mem2 -->|Zero-Copy Write| E[(Gold: Parquet Tables)]
        T3(DuckDB SQL) -.-> Mem2
        E --> F[Gold Data Contract]
        class E storage
        class Mem2 memory
        class F qa
        class T3 tool
        
        %% Downstream
        F -->|Consumption| Split{Readiness}
        Split --> G[(Machine Learning & MLflow)]
        Split --> H[[BI Dashboards & Analytics]]
        class G compute
        class H compute
    end

    %% Delta Time Travel
    DeltaLog(JSON _delta_log Ledger)
    TimeTravel((Time Travel & Rollback <br> VERSION AS OF X))
    class DeltaLog,TimeTravel delta_feature
    B -.->|Updates| DeltaLog
    C -.->|Updates| DeltaLog
    DeltaLog -.-> TimeTravel

    %% Remote Storage
    RemoteStorage[(DagsHub / S3 Object Storage)]
    class RemoteStorage remote
    
    B -.->|dvc push| RemoteStorage
    C -.->|dvc push| RemoteStorage
    E -.->|dvc push| RemoteStorage

    %% GE Reference Node
    GX([Great Expectations <br> HTML Data Docs UI])
    class GX tool
    D -.-> GX
    F -.-> GX

    %% DVC Caching Explanation Nodes
    Z1[DVC hashes inputs] -..-> A
    Z2[If unchanged: Skip extraction] -..-> B
    class Z1,Z2 dvc_stage
```

## The Single "Data Lakehouse" Ecosystem
Traditionally, data copied from a messy Data Lake into an expensive, separate Data Warehouse. StoreCast abandons this by leveraging a single **Data Lakehouse**. The entire system exists on local files, logically separated into three maturity layers:

- **Bronze (Raw Ingestion):** The immutable historical record. PySpark faithfully dumps the external CSVs into robust `Delta` folders but makes no attempt to clean them. 
- **Silver (Cleansed Source of Truth):** The enterprise anchor layer. Polars applies strict schema enforcement, clipping, and anomaly resolution. Because it is a `Delta` table, it retains full ACID logic and Time Travel if data corruption occurs.
- **Gold (Aggregated / OBT):** We intentionally abandon both `Delta` format and complex `Star Schemas` here. Parquet does not have Time Travel, and we do not use Fact/Dimension tables. Instead, DuckDB flattens everything into a highly aggregated **One Big Table (OBT)** (`master_sales.parquet`). This simple Parquet file is optimized strictly for maximal read-speed, designed to be ingested flawlessly by our Machine Learning models and BI Dashboards without requiring them to join anything.

## The "Tri-Stack" Tooling Rationale

Rather than defaulting to `pandas` or adopting heavyweight infrastructure like `dbt`, StoreCast uses a highly specialized "Tri-Stack" approach to data processing, adhering strictly to the "use the right tool for the job" mantra.

### 1. PySpark (Bronze Ingestion)
- **The Phase:** Raw data extract and append.
- **The Justification:** While Pandas must hold entire datasets in RAM, PySpark is a distributed engine that lazy-evaluates data. In our pipeline, PySpark effortlessly streams raw retail CSVs, inferring massive unstructured schemas and writing them out as compressed `Delta Lake` tables with ACID transaction guarantees. This proves our architecture can theoretically handle Petabyte-scale ingestion without rewriting code.

### 2. Polars (Silver Transformation)
- **The Phase:** Cleaning anomalies, deduplicating, and standardizing datatypes.
- **The Justification:** Polars is a blazingly fast DataFrame library written in Rust. Because Silver transformations (clipping negative sales, dropping duplicates, mapping types) are largely row-level operations, they are memory-bound. Polars evaluates queries natively using multi-threading and SIMD (Single Instruction, Multiple Data), mathematically outperforming PySpark overhead on single-node hardware.

### 3. DuckDB (Gold Modeling & Feature Engineering)
- **The Phase:** Analytics flattening, deep time-series window functions.
- **The Justification:** Creating ML features (like a 52-week time-travel lag and a 4-week moving average over 4,500 stores simultaneously) via Pandas `.groupby().shift()` is highly prone to data leakage and massive memory spikes. **DuckDB** executes native vector-based zero-copy SQL directly against our Silver Delta tables. It flawlessly handles massive `LEFT JOINs` and `WINDOW PARTITION BY` analytical queries in fractions of a second, completely circumventing the need for expensive tools like `dbt` or Snowflake.

## Cloud Scalability Parity
While this current architecture is engineered to execute locally with a **$0 development footprint**, it is designed for perfect 1:1 cloud production parity:
- **PySpark:** Natively deploys to AWS EMR or Databricks for massive horizontal scaling if the data grows into Petabytes.
- **Polars & DuckDB:** Easily containerized into Docker, running seamlessly inside cheap AWS Fargate instances or Kubernetes pods, executing the exact same zero-copy SQL against an S3 bucket without requiring any code refactoring.

## The Quality Gates (Great Expectations)
A Medallion Lakehouse is useless if the data is poisoned. 
Between the Silver and Gold layers, we enforce strict **Data Contracts** mapped by Great Expectations (GX).

- **Silver Gates:** Ensure primary keys (`Store`, `Dept`, `Date`) are mathematically unique (no cartesian explosion duplicates) and sales constraints (no negative numbers) are held.
- **Gold Gates:** Ensure our advanced DuckDB features successfully mapped without returning catastrophic `NULL` explosions (except for expected mathematical warm-up periods). 
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
- **Zero-Copy Remote ELT:** In development, dashboards query the local Parquet files. However, in production, engines like DuckDB use HTTP Range Requests to query the remote DagsHub Parquet files directly. DuckDB reads the Parquet metadata footer, identifies exactly where the necessary data lives on the remote server, and downloads only those specific chunks over the network. This provides blazing fast, zero-copy cloud ELT!
