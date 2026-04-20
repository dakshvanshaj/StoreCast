# Architectural Trade-off: The Forecasting Horizon

In any Time-Series Machine Learning system, the most critical architectural constraint is the **Forecasting Horizon** (how far into the future the algorithm can accurately predict).

## Current Architecture Horizon: 1 Week
In the StoreCast baseline, we utilize `rolling_mean_4` (the mathematical average of the preceding 4 weeks of sales) as an autoregressive anchor feature. 

Because of this specific engineered feature, our Absolute Forecast Horizon is **1 week**.
* **Predicting `t+1`**: We have actual, observed historical sales data to calculate the `rolling_mean_4` for `t`.
* **Predicting `t+2`**: We do *not* have actual historical sales for `t+1`. Our rolling mean equation is missing 25% of its required data.

### The Solution: Recursive Forecasting
To predict `t+2`, `t+3`, or `t+4` (a full month out), we must implement a **Recursive Forecasting Loop**.
1. We predict `t+1` using actual historical data.
2. We artificially inject the *predicted* dollar value for `t+1` back into the DataFrame as if it actually happened.
3. We recalculate the `rolling_mean_4` using the simulated prediction.
4. We predict `t+2` using the simulated mean.

**The Trade-off:** By feeding predictions back into the feature table, the inherent model error compounds recursively. While `t+1` might have a 7% WMAPE, `t+4` could degrade to a 15% WMAPE due to the cascading drift of relying on simulated data.

## Alternative Architecture: Direct Forecasting (52-Week Horizon)
If the business demanded a highly stable, 12-month forward-looking estimate rather than a hyper-accurate next-week estimate, we would pivot the architecture.

By mathematically dropping `rolling_mean_4` and relying exclusively on deep historical anchors (like `lag_52` or `lag_104`), we unlock **Direct Forecasting**.
* To predict Week 144, the model looks at Week 92.
* To predict Week 180, the model looks at Week 128.

Because we already possess the deep historical truth for `lag_52`, we never have to simulate data. We can predict 52 individual weeks simultaneously with an identical mathematical baseline error. 

**The Trade-off:** Dropping the short-term 4-week mean strips the model of recent momentum contextualization (e.g., if a store had a massive anomaly or growth explosion in Weeks 140-143, a `lag_52` exclusive model would be completely blind to it, relying solely on last year's seasonality). 

## Executive Summary
StoreCast prioritizes short-term accuracy to reduce immediate inventory safety stock. We accept the compounding error of Recursive Forecasting at `t+4` in exchange for maximizing the accuracy at `t+1`.
