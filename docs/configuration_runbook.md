# System Configuration & Execution Runbook

This document tracks the precise commands required to reproduce the StoreCast pipeline. As a lean, fast, production-grade project, we prioritize single-command reproducibility and simple dependency management via `uv`.

## 1. Environment & Packaging
**Tool**: `uv` (Fast Python package manager)

```bash
# Sync all dependencies exactly as locked
uv sync

# Install the project locally so internal modules resolve automatically
uv pip install -e .
```

## 2. Baseline Model Execution
**Task**: Calculate the LYSW manual forecasting error metrics.

```bash
uv run python baseline.py
```

## 3. Data Pipeline (Medallion Architecture)
**Task**: Run the ETL jobs sequentially.

### Bronze Ingestion
Extracts `data/raw` CSVs and partitions them into `data/bronze` Delta Tables using PySpark.
```bash
uv run python -m data.ingest_bronze
```

### Silver Transformation (Pending)
Cleans the Bronze Delta tables using Polars.
```bash
uv run python -m data.clean_silver
```

## 4. Documentation Server
**Tool**: `mkdocs`
**Task**: Serve this site locally with live-reload.

```bash
mkdocs serve
```
