# System Configuration & Execution Runbook

This document tracks the precise commands required to reproduce the StoreCast pipeline. As a lean, fast, production-grade project, we prioritize single-command reproducibility and simple dependency management via `uv`.

## 1. Environment & Packaging
**Tool**: `uv` (Fast Python package manager)

```bash
# Sync all dependencies exactly as locked
uv sync

# Add required MLOps modules specifically needed for Phase 4
uv add optuna dagshub

# Install the project locally so internal modules resolve automatically
uv pip install -e .
```

## 2. Baseline Model Execution
**Task**: Calculate the LYSW manual forecasting error metrics.

```bash
uv run python baseline.py
```

## 2b. MLOps Hyperparameter Optimization & Model Registry
**Task**: Run the Bayesian Search grid and log tracing metrics remotely to DagsHub.
```bash
uv run python -m src.models.optimizer
```

## 2c. Enterprise Champion Deployment
**Task**: Isolate the top candidate from the registry via multi-objective Pareto filtering, retrain seamlessly on the full unified dataset, and rigorously evaluate final latency before formally promoting the model.
```bash
uv run python -m src.models.deploy_champion
```

## 2d. SHAP Model Explainability
**Task**: Rip the decision tree weights from the serverless `@production` model and generate Game-Theoretic visual analytics to `docs/images/shap/`.
```bash
uv run python -m src.models.explainability
```

## 2e. Operational Batch Inference
**Task**: Run the live Batch Inference engine to dynamically score future data arrays, outputting the final business forecast table.
```bash
uv run python -m src.models.batch_inference
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
