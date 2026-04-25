# Advanced Analytics: Store Segmentation Clustering

Another critical pillar of the StoreCast project is the ability to personalize marketing and markdown strategies. Forecasting tells us *how much* to ship to a store, but it does not tell us *how to treat* that store from a promotional perspective.

This document outlines the business context and mathematical logic behind the Unsupervised Store Clustering pipeline (`src/models/store_clustering.py`).

---

## The Business Goal
The 45 stores in our pilot region are not identical. Sending a $50,000 Markdown budget to a tiny neighborhood express store is wasted margin because they lack the physical floor space to support the foot traffic. We need to group our 45 stores into strategic "Archetypes" based on their physical scale and behavioral response to markdowns so Marketing can execute targeted campaigns.

## The Algorithm
We utilize Scikit-Learn's **`K-Means Clustering`**. 

1. **Feature Aggregation:** We collapse the Gold Data into exactly 45 rows (one per store), calculating the `avg_weekly_sales`, `store_size`, and `avg_weekly_markdown` across all time.
2. **Standardization:** K-Means calculates the physical Euclidean distance between data points. We must use `StandardScaler` to normalize the data; otherwise, `store_size` (200,000 sq ft) would mathematically overpower `avg_weekly_sales` ($15,000) simply because the absolute integers are larger.
3. **Clustering:** The algorithm assigns each store to a cluster.

## The Mathematical Optimization: The Elbow Method
Because K-Means is unsupervised, it requires the engineer to explicitly state how many clusters (`K`) to build. 

To find the mathematically optimal number of clusters, we utilized the **Elbow Method**:
- We tested K from 1 to 10 and plotted the **Inertia** (Within-Cluster Sum of Squares / Variance).
- The variance crashed by ~90 points from K=1 to K=2, and another ~25 points from K=2 to K=3. 
- However, from K=3 to K=4, the variance only dropped by 4 points before completely flatlining.
- The mathematical "Elbow" (point of diminishing returns) is exactly **K=3**.

## Enterprise Retail Taxonomy Mapping
By enforcing K=3, the algorithm organically recreated standard enterprise retail taxonomy. Based on the cluster centers, we mapped the raw integer outputs to the following actionable business archetypes:

1. **Supercenter (Flagships)** [Cluster 2]
   - *Profile:* ~195k Sq Ft | ~$21k Weekly Dept Sales | ~$9.4k Markdowns
   - *Strategy:* Anchor stores. Focus on volume retention and brand protection. Keep shelves stocked to avoid massive revenue bleeding.
2. **Standard Discount Store** [Cluster 0]
   - *Profile:* ~128k Sq Ft | ~$11.8k Weekly Dept Sales | ~$6.5k Markdowns
   - *Strategy:* The "Middle Class". Ideal testing ground for Market Basket A/B tests before rolling them out to the Flagships.
3. **Neighborhood Market (Express)** [Cluster 1]
   - *Profile:* ~47k Sq Ft | ~$6.6k Weekly Dept Sales | ~$1.7k Markdowns
   - *Strategy:* High-margin essentials. Do not waste markdown promotional budgets here due to low footprint capacity and low baseline responsiveness.

## Segmentation Quality Validation
We evaluated the final 3-cluster architecture using the **Silhouette Score**. Our model achieved a score of **0.521**, proving the clusters are highly dense and mathematically distinct from one another, crossing the required threshold for production business utility.
