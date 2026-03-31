# StoreCast: Production-Grade Retail Demand Forecasting

StoreCast is a full-stack, enterprise-grade data engineering and automated machine learning pipeline. Designed for a 45-store retail pilot, the system targets a **reduction of $17.3M in trapped working capital** by replacing manual "Last-Year-Same-Week" heuristics with machine learning algorithms.

---

## 1. Platform Architecture
StoreCast simulates an advanced Databricks-style **Lakehouse** entirely on-premise using the Medallion Architecture:

*   **Bronze Layer (Ingestion):** We use **PySpark** configured with **Delta Lake** to extract raw 143-week CSVs and write them as partitioned, ACID-compliant Delta tables.
*   **Silver Layer (Transformation):** We use **Polars** (the blazing-fast Rust engine) to process Data Analyst EDA rules (e.g., clipping negative sales returns, forward-filling macro-economic lag variables).
*   **Gold Layer (Modeling):** We use **DuckDB** to execute rapid in-process SQL aggregations to build our Star Schema.
*   **BI & Machine Learning:** Gold tables feed directly into PowerBI dashboards and XGBoost forecasting models.

## 2. The Business Problem
Currently, our demand planners rely on manual forecasting, resulting in an 11.85% WMAPE error rate. This causes:
1. **Safety Stock Bloat:** Tying up massive amounts of working capital.
2. **Untargeted Markdown Bleed:** Applying promotions blanketly without evaluating local elasticity. 

All math and baseline metrics are transparently documented via MkDocs.

## 3. Project Structure
```text
/
├── data/               # Local Delta Lakehouse (raw/, bronze/, silver/, gold/)
├── docs/               # MkDocs source (Business contexts, runbooks)
├── notebooks/          # Targeted EDA & ML Experimentation
├── src/                # Python Package (Data Pipelines & Modeling)
├── pyproject.toml      # Dependency management via uv
└── mkdocs.yaml         # Documentation config
```

## 4. How to Run
#### Setup
1. Sync Dependencies and install the package locally:
```bash
uv sync
uv pip install -e .
```

#### Pipeline Execution
View the `docs/configuration_runbook.md` for exact line-by-line commands for executing the Bronze and Silver pipelines (e.g., `uv run python -m src.data.ingest_bronze`).

#### Documentation
Start the local MkDocs server to view Business KPIs, EDA rendering, and system specs:
```bash
mkdocs serve
```
