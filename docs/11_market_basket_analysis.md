# Advanced Analytics: Market Basket Analysis

Another critical pillar of the $16M ROI case is the Market Basket Analysis pipeline (`src/analytics/market_basket.py`).

This document outlines the business context, algorithmic choices, and the critical mathematical fixes implemented to ensure this model provides true causal value rather than "Academic Noise".

---

## The Business Goal
Marketing directors need to know which departments are highly linked to execute **Loss-Leader Pricing Strategies**. If we know that discounting Department A reliably drives foot traffic to Department B, we can sacrifice margin on A to generate massive holistic profit on B.

## The Algorithm
True Market Basket Analysis (like Apriori or FP-Growth) requires Receipt-Level data (scanning individual shopping carts). Because we only have Aggregated Weekly Data at the department level, we must use a statistical proxy: The **Pearson Correlation Matrix**.

We pivot the Gold data so every Department becomes a column. The `.corr()` algorithm then calculates the linear relationship between the columns: *"If Dept A has a great week, does Dept B also have a great week?"*

## The Architectural Trap: Confounding Seasonality
Initially, running Pearson Correlation on raw `weekly_sales` yielded heavily biased results. 

This is the famous statistical axiom: **Correlation does not imply Causation.** 
If we run raw correlations, Turkeys and Televisions will show a massive `0.95` correlation. Why? Because they both spike in Week 47 (Black Friday). They are not actually linked to each other; they are both responding to a third Confounding Variable: **Seasonality/Holidays**. Discounting Turkeys in March will not cause a spike in Televisions.

## The Fix: De-Seasonalized Residual Correlation (Partial Correlation)
To strip the Holiday Bias out of the mathematics, we upgraded the algorithm to analyze **Residuals** (Surprises) rather than absolute dollars:

1. **Calculate the Seasonal Baseline:** We extract `sales_last_year` directly from DuckDB to act as our expected baseline.
2. **Calculate the Residual:** `residual_sales = weekly_sales - sales_last_year`. (If Turkeys spike exactly as much as we expected them to on Thanksgiving, the Residual is $0).
3. **Run Pearson on the Surprises:** We pivot the dataframe using the `residual_sales` instead of `weekly_sales`. 

By correlating the residuals, the algorithm completely ignores expected Holiday spikes. It only flags a relationship if two departments experience an *unexpected* surge simultaneously. This rigorously isolates true causal affinity and provides Marketing with bulletproof, deduplicated Markdown pairings.
