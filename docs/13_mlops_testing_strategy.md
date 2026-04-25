# MLOps Testing Strategy & Coverage

In traditional software engineering, the standard goal is to achieve 90%+ total test coverage across the entire codebase. However, in **Data Engineering and MLOps**, striving for 100% code test coverage is an anti-pattern that leads to bloated, brittle pipelines. 

This document outlines the official StoreCast testing philosophy, detailing exactly what we test, what we intentionally ignore, and how our PyTest suite ensures mathematical stability without wasting engineering hours.

---

## 1. What We DO NOT Test (0% Coverage is Expected)

### A. Data Orchestration & SQL Execution
We do not write unit tests for files like `src/data/clean_silver.py` or `src/data/create_gold.py`.
- **The Logic:** These scripts are predominantly executing DuckDB SQL queries (e.g., `LEFT JOIN`) or Polars memory manipulations. We do not need to write a PyTest to verify that DuckDB knows how to join tables.
- **The Alternative:** Instead of testing the *code*, we test the *output data*. This is handled by **Data Quality Validation** (Great Expectations in `validate_silver.py`), which asserts that the output schema is correct, nulls are bounded, and foreign keys exist.

### B. External ML Frameworks
We do not write unit tests for `.fit()` or `.predict()` calls inside `baseline.py` or `trainer.py`.
- **The Logic:** Scikit-Learn and XGBoost are heavily tested open-source libraries. We do not need to write a unit test to prove that an XGBoost Regressor can execute a mathematical tree split. 

---

## 2. What We MUST Test (Aim for 80%+ Coverage)

We enforce strict PyTest coverage on all **Custom Mathematics**, **Data Boundaries**, and **Unsupervised Logical Assumptions**. If these components break, they will not throw a Python error; they will silently corrupt the data and inflate the model's perceived accuracy.

### A. Chronological Splitting (Preventing Data Leakage)
**Script:** `tests/test_data_pipeline.py`
- **The Threat:** If `prepare_data()` accidentally shuffles data or miscalculates quantiles, we will leak future sales data into the training set. The model will appear to have a 1% error rate during training, but will fail catastrophically in production.
- **The Test:** We mock a synthetic DataFrame using `@patch` to bypass disk I/O. We mathematically assert that the 70/15/15 split executes perfectly, and that the chronological dates do not overlap between Train, Validation, and Test matrices.

### B. Custom Business Metrics
**Script:** `tests/test_metrics.py`
- **The Threat:** The CFO mandated that errors on Holiday weeks cost the business 5x more than errors on normal weeks. If our `wmape_metric` multiplier math is flawed, the Bayesian Optimizer will promote models that optimize for the wrong weeks.
- **The Test:** We pass identical absolute errors to the metric calculator—one on a normal week, and one on a holiday week. We assert that the final WMAPE percentage correctly inflates when the `is_holiday` flag is triggered.

### C. Unsupervised Analytical Ratios
**Script:** `tests/test_advanced_analytics.py`
- **The Threat:** Isolation Forests and Pearson matrices are vulnerable to absolute volume biases (e.g., massive stores always appearing anomalous simply because their revenue is high).
- **The Test:** We assert that `anomaly_detection.py` successfully converts raw dollars into dimensionless `yoy_growth_ratios` (compressing massive stable stores to `1.0`), and we assert that `market_basket.py` correctly subtracts `sales_last_year` to calculate true `residual_sales` before running correlations.

---

## Running the Test Suite
To execute the test suite and verify the mathematical integrity of the StoreCast pipeline, run:
```bash
uv run pytest tests/ -v
```

To run a coverage sweep and ensure critical boundary logic is covered, run:
```bash
uv run pytest --cov=src tests/
```
