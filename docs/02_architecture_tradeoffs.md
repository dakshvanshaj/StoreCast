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

## Summary of the Tri-Stack Philosophy:
- **PySpark (Bronze):** Maximizes horizontal distributed scale for raw unbounded data extraction.
- **Polars/DuckDB (Silver/Gold):** Maximizes vertical speed and minimizes cloud computing costs for analytical processing and feature engineering on structured, partitioned data.
