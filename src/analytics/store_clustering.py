import polars as pl
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import structlog
from src.utils.config_manager import ConfigManager
from pathlib import Path

logger = structlog.get_logger()

def compute_store_clusters():
    """
    Executes Unsupervised K-Means Clustering on the 45 stores.
    Groups stores into "Archetypes" based on Volume, Size, and Markdown behavior.
    """
    logger.info("Initializing Store Segmentation via K-Means...")
    
    # 1. Aggregate Features per Store from Gold Data
    # We lazily scan the Gold dataset and collapse 470,000 rows into exactly 45 rows (one per store)
    cfg = ConfigManager()
    lf = pl.scan_parquet(cfg.get("data.paths.gold_data"))
    
    store_profiles = lf.group_by("store").agg([
        pl.col("weekly_sales").mean().alias("avg_weekly_sales"),
        pl.col("store_size").first().alias("store_size"),
        # We fill nulls with 0 for markdowns before averaging to get true responsiveness
        pl.col("total_markdown").fill_null(0).mean().alias("avg_weekly_markdown")
    ]).collect().to_pandas()
    
    # 2. Standardize Features
    # K-Means calculates physical distance between points. If we don't scale, 
    # 'store_size' (200,000) will overpower 'avg_weekly_sales' ($15,000) simply because the raw number is larger.
    scaler = StandardScaler()
    features_to_scale = store_profiles.drop("store", axis=1)
    X_scaled = scaler.fit_transform(features_to_scale)
    
    # 3. Fit K-Means
    logger.info("Fitting K-Means algorithm (K=4)...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    store_profiles['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Evaluate Segmentation Quality
    # Silhouette Score ranges from -1 (terrible) to +1 (perfectly separated dense clusters)
    score = silhouette_score(X_scaled, store_profiles['Cluster'])
    logger.info("K-Means clustering complete!", num_clusters=4, silhouette_score=round(score, 3))
    
    # Let's log the center of each cluster so we can understand the "Archetypes"
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features_to_scale.columns)
    cluster_centers.index.name = 'Cluster'
    logger.info("Cluster Archetype Centers:")
    print(cluster_centers.round(0))
    
    # 5. Map mathematical clusters to Corporate Retail Taxonomy
    # Based on the K=3 cluster centers:
    taxonomy_mapping = {
        2: "Supercenter (Flagship)",
        0: "Standard Discount Store",
        1: "Neighborhood Market (Express)"
    }
    store_profiles['Archetype'] = store_profiles['Cluster'].map(taxonomy_mapping)
    
    # 6. Export for Dashboard Consumption
    out_path = Path(cfg.get("data.paths.store_clusters_export"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    store_profiles.to_csv(str(out_path), index=False)
    
    logger.info("Store Clusters exported successfully!", path=str(out_path))

if __name__ == "__main__":
    compute_store_clusters()
