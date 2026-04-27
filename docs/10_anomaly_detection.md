# Advanced Analytics: Context-Aware Anomaly Detection

While the core of the StoreCast project revolves around Supervised Demand Forecasting (XGBoost), the $16M ROI case is heavily supported by auxiliary "Data Products". One of these is the Unsupervised Anomaly Detection pipeline (`src/analytics/anomaly_detection.py`).

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
- `markdown_intensity_ratio = total_markdown / weekly_sales`

By feeding the Isolation Forest these dimensionless ratios, a massive Grocery department operating normally has a ratio of `1.0`. The algorithm completely ignores its massive absolute dollar volume. Instead, the tree is mathematically forced to hunt for pure **Operational Decoupling** (e.g., a store that normally makes $10k suddenly making $0, resulting in an anomalous ratio of `0.0`).

## Advanced Contextual Features
To make the model even more lethal at catching precise business failures, we engineered three additional advanced ratios into the Polars ingestion layer:

### 1. Markdown Cannibalization
- **Math:** `markdown_intensity_ratio / (yoy_growth_ratio + 0.01)`
- **The Flag:** If you heavily discount items, your Sales Growth *should* explode upwards. If a store has a massive Markdown Intensity (20%), but their YoY Growth is `0.8` (they are shrinking), this flags a catastrophic anomaly. It means Marketing is giving away products for free, and the store is *still* losing money compared to last year. It flags completely ineffective promotions.

### 2. Footprint Yield (Real-Estate Bloat)
- **Math:** `weekly_sales / store_size`
- **The Flag:** A massive 200,000 sq ft Supercenter selling $50,000 a week is actually performing *terribly* (Yield: $0.25 per sq ft). A tiny 40,000 sq ft Express store selling $40,000 is performing *amazingly* (Yield: $1.00 per sq ft). Tracking a sudden drop in this ratio isolates wasted floor space and bloated real-estate.

### 3. Holiday Miss Severity (Supply Chain Failure)
- **Math:** Flagging weeks where `is_holiday == 1` AND `trend_deviation_ratio < 1.0`.
- **The Flag:** During Thanksgiving or Christmas, sales should inherently spike above the 4-week rolling average. If a holiday occurs, and a department's trend ratio is `0.9` (they actually sold *less* than the previous boring weeks), that is a massive anomaly. It usually means a supply chain truck didn't arrive and the shelves were empty during peak rush!
