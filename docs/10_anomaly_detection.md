# Advanced Analytics: Context-Aware Anomaly Detection

While the core of the StoreCast project revolves around Supervised Demand Forecasting (XGBoost), the $16M ROI case is heavily supported by auxiliary "Data Products". One of these is the Unsupervised Anomaly Detection pipeline (`src/models/anomaly_detection.py`).

This document outlines the business context, algorithmic choices, and the critical mathematical fixes implemented to ensure this model provides true enterprise value rather than "Academic Noise".

---

## The Business Goal
Retail operations are messy. A roof leak might cause a department to close, or an unannounced competitor promotion might crush sales. Stakeholders need to be alerted when a department drastically decouples from its expected operational trend so they can investigate immediately.

## The Algorithm
We utilized Scikit-Learn's **`IsolationForest`**. Isolation Forests work by randomly selecting a feature and randomly splitting the dataset. Data points that require very few splits to isolate are deemed "anomalous" because they are mathematically far away from the dense clusters of normal data.

## The Architectural Trap: Volume Outliers
Initially, we fed the algorithm absolute raw dollars (`weekly_sales`, `sales_last_year`, `rolling_4_wk_sales_avg`). 

This triggered a classic trap: **Alert Fatigue caused by Volume Outliers.** 
In retail, a Grocery department might regularly sell $150,000 a week, while an Electronics department sells $10,000. Because $150,000 is in the 99th percentile of all sales volume across the entire company, the Isolation Forest simply drew a split at `weekly_sales > $100,000` and immediately isolated the Grocery departments. 

It blindly flagged them as "Anomalies" *every single week* simply because they were large, completely ignoring the fact that their historical lags proved that $150,000 was perfectly normal for them.

## The Fix: Dimensionless Ratios
To make the algorithm truly "Context-Aware", we intercepted the data using Polars and converted the raw dollars into mathematical Ratios:
- `yoy_growth_ratio = weekly_sales / sales_last_year`
- `trend_deviation_ratio = weekly_sales / rolling_4_wk_sales_avg`

By feeding the Isolation Forest these dimensionless ratios, a massive Grocery department operating normally has a ratio of `1.0`. The algorithm completely ignores its massive absolute dollar volume. Instead, the tree is mathematically forced to hunt for pure **Operational Decoupling** (e.g., a store that normally makes $10k suddenly making $0, resulting in an anomalous ratio of `0.0`).
