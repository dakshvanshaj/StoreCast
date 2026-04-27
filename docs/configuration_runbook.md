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

## 2. Configuration
**Tool**: `config/params.yaml` + `src/utils/config_manager.py`

All project constants (data paths, feature lists, split quantiles, hyperparameters, quality gates, experiment names, observability thresholds) are centralized in a single YAML file. No hardcoded Python config modules exist.

```bash
# Verify ConfigManager loads correctly
uv run python -c "from src.utils.config_manager import ConfigManager; cfg = ConfigManager(); print(cfg.get('project.name'))"
```

## 3. Data Pipeline (Medallion Architecture)
**Task**: Run the ETL jobs sequentially.

### Bronze Ingestion
Extracts `data/raw` CSVs and partitions them into `data/bronze` Delta Tables using PySpark.
```bash
uv run python -m src.data.ingest_bronze
```

### Silver Transformation
Cleans the Bronze Delta tables using Polars (streaming mode).
```bash
uv run python -m src.data.clean_silver
```

### Silver Validation (Great Expectations)
```bash
uv run python -m src.data.validate_silver
```

### Gold Feature Engineering (DuckDB)
Computes temporal Window Functions, Lags, Rolling Averages via DuckDB SQL.
```bash
uv run python -m src.data.create_gold
```

### Gold Validation (Great Expectations)
```bash
uv run python -m src.data.validate_gold
```

### Chronological ML Split
Generates static `train/val/test` parquet splits from Gold data.
```bash
uv run python -m src.data.chronological_split
```

## 4. Baseline Model Execution
**Task**: Calculate the LYSW manual forecasting error metrics.
```bash
uv run python -m src.baseline.baseline
```

## 5. Model Training & Optimization

### Baseline Random Forest
```bash
uv run python -m src.training.baseline_rf
```

### Baseline Trainer (Linear Regression + XGBoost)
```bash
uv run python -m src.training.trainer
```

### Bayesian Hyperparameter Optimization (Optuna + MLflow)
```bash
uv run python -m src.training.optimizer
```

## 6. Deployment Pipeline

### Stage Candidate (Pareto Selection → Retrain → @candidate tag)
```bash
uv run python -m src.deployment.stage_candidate
```

### Promote Champion (Human-in-the-Loop approval: @candidate → @production)
```bash
uv run python -m src.deployment.promote_champion
```

### Operational Batch Inference
```bash
uv run python -m src.deployment.batch_inference
```

## 7. Advanced Analytics

### Context-Aware Anomaly Detection
```bash
uv run python -m src.analytics.anomaly_detection
```

### Market Basket Analysis
```bash
uv run python -m src.analytics.market_basket
```

### Store Segmentation (K-Means Clustering)
```bash
uv run python -m src.analytics.store_clustering
```

## 8. Observability

### SHAP Model Explainability
```bash
uv run python -m src.observability.explainability
```

### Drift Monitoring (Evidently AI)
```bash
uv run python -m src.observability.drift_monitor
# Launch Evidently dashboard
uv run evidently ui
```

## 9. Business Dashboard
```bash
uv run streamlit run src/app_business.py
```

## 10. Tests
```bash
uv run pytest tests/ -v
```

## 11. Documentation Server
**Tool**: `mkdocs`
```bash
mkdocs serve
```
