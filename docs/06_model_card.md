# ML Model Card: StoreCast Demand Forecasting

## 1. Model Details
- **Architecture Base:** Extreme Gradient Boosting (`XGBRegressor`)
- **Wrapper Construction:** Natively wrapped in Scikit-Learn's `TransformedTargetRegressor` bridging inputs through `np.log1p` and predictions through `np.expm1`.
- **Primary Use Case:** Forecasting granular, department-level weekly retail demand across an anonymized 45-store network.
- **Tracking & Governance:** All runs, schemas, and signatures inherently locked into a centralized DagsHub MLflow ledger.

## 2. Intended Use and Business Context
- **Business KPI:** Driving targeted reductions in physical Inventory Working Capital via the minimization of Safety Stock buffering.
- **Financial Baseline:** Original "Last-Year-Same-Week" (LYSW) manual baseline yielded an **11.85% WMAPE**, necessitating roughly $216M in standing safety stock.
- **Model Target:** Decrease global un-weighted predictive error to shrink overall safety margins by 9.5% without jeopardizing localized Fill Rates.

## 3. Training & Evaluation Data (Chronological Matrix)
Because Retail Forecasting demands high protection against data leakage, K-Fold cross-validation was algorithmically rejected. The dataset (~476,000 distinct time-series nodes) was strictly separated across a temporal domain:
- **Train Set (70%):** Weeks 1–100.
- **Validation Set (15%):** Weeks 101–121 (Primary benchmark for Bayesian Search Grid).
- **Test / Holdout Set (15%):** Weeks 122–143.

## 4. Hyperparameter Search Grid (Optuna)
Instead of exhausting expensive cloud compute with an unguided GridSearch, StoreCast executed a bounded Bayesian Search spanning 50 trial epochs:

| Parameter | Type | Distribution Bounds | Optuna Space |
| :--- | :--- | :--- | :--- |
| `n_estimators` | Integer | [100, 1000] | `suggest_int('n_estimators', 100, 1000)` |
| `learning_rate` | Float (Log) | [0.01, 0.3] | `suggest_float('learning_rate', 0.01, 0.3, log=True)` |
| `max_depth` | Integer | [3, 10] | `suggest_int('max_depth', 3, 10)`|
| `subsample` | Float | [0.5, 1.0] | `suggest_float('subsample', 0.5, 1.0)` |
| `colsample_bytree` | Float | [0.5, 1.0] | `suggest_float('colsample_bytree', 0.5, 1.0)`|

fANOVA visual analytics proved that the `learning_rate` controlled absolute convergence momentum, while `max_depth` yielded negligible variance shifts.

## 5. Final Model Geometry & Quantitative Analysis
Upon convergence observation (plateauing roughly at Trial 15), the MLflow ledger registered the optimal configuration resulting in a record minimum predictive error.

**Operational Parameters:**
- `n_estimators`: 610
- `learning_rate`: 0.0153
- `max_depth`: 10
- `subsample`: 0.686
- `colsample_bytree`: 0.864

**Financial Output Matrix:**
- **Final Validation WMAPE:** **7.76%**
- **Absolute Gain over LYSW Matrix:** 4.09%
- **Simulated Capital Release Projection:** **$20.53M**

## 6. Known Limitations
- The model exhibits dependency on the 52-week lag metric (`lag_52`). New store locations (lacking 1 year of historical operation data) cannot inherently leverage this architecture and must regress to secondary zero-start models or regional average bootstrapping.
- Because of categorical logic limits, predictions cannot currently cross-pollinate out-of-boundary departments dynamically without resorting to separate unsupervised cluster models (Reference: Phase 3.5 Market Basket).
