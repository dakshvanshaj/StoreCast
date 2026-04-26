import polars as pl
import pandas as pd
import structlog
from src.utils.config_manager import ConfigManager
from pathlib import Path

logger = structlog.get_logger()

def compute_market_basket() -> None:
    """
    Executes an Unsupervised Market Basket Analysis using Pearson Correlation.
    Identifies high-confidence purchasing affinities across departments (e.g. If Dept A spikes, Dept B spikes).
    """
    logger.info("Initializing Market Basket Apriori-style Clustering...")
    
    # 1. Use Polars to lazily scan the correct Gold Parquet.
    # We extract 'sales_last_year' to act as our mathematically proven Expected Seasonal Baseline!
    cfg = ConfigManager()
    lf = pl.scan_parquet(cfg.get("data.paths.gold_data")).select(['store', 'date', 'dept', 'weekly_sales', 'sales_last_year'])
    
    # Drop the first 52-weeks of warm-up data where the historical baseline doesn't exist yet
    lf = lf.drop_nulls(subset=["sales_last_year"])
    
    # 2. De-Seasonalize the data! (The Surprise Factor)
    # Residual = Actual Sales - Expected Seasonal Sales
    # We are mathematically isolating the "Surprise Spikes" and stripping away the Holiday bias.
    lf = lf.with_columns(
        (pl.col("weekly_sales") - pl.col("sales_last_year")).alias("residual_sales")
    )
    
    logger.info("Pivoting transactional data into Department Basket vectors...")
    df_collected = lf.collect()
    
    # Cast 'dept' to string so the pivot creates clean string column names
    df_collected = df_collected.with_columns(pl.col('dept').cast(pl.String))
    
    # 3. Pivot in Polars (happens in Rust for extreme memory efficiency)
    # CRITICAL: We pivot on the 'residual_sales' to hunt for pure affinity, NOT 'weekly_sales'
    basket = df_collected.pivot(
        values='residual_sales', 
        index=['store', 'date'], 
        on='dept',
        aggregate_function='first'
    ).fill_null(0)
    
    logger.info(f"head() of pivoted basket: {basket.head()}")
    
    logger.info("Calculating Pearson Affinity Matrix...")
    # Polars doesn't have a native 2D correlation matrix function.
    # We drop the index columns, then cast the isolated feature columns to Pandas to leverage its highly optimized C-backed .corr()
    dept_matrix = basket.drop(['store', 'date']).to_pandas()
    correlation_matrix = dept_matrix.corr()
    
    logger.info(f"head() of correlation matrix: {correlation_matrix.head()}")
    logger.info("Isolating strict associative rules (Correlation > 0.6)...")
    pairs = correlation_matrix.unstack().reset_index()
    pairs.columns = ['Department_A', 'Department_B', 'Correlation_Score']
    
    logger.info(f"head() of pairs: {pairs.head()}")
    
    # Convert back to integers for clean logic
    pairs['Department_A'] = pairs['Department_A'].astype(int)
    pairs['Department_B'] = pairs['Department_B'].astype(int)
    
    # Drop self-correlations AND mirror duplicates (A->B is the same as B->A)
    # By strictly enforcing A < B, we eliminate the symmetric redundancy!
    pairs = pairs[pairs['Department_A'] < pairs['Department_B']]
    
    # Filter for high-confidence rules
    strong_rules = pairs[pairs['Correlation_Score'] > 0.6].sort_values('Correlation_Score', ascending=False)
    
    # Push to Advanced Analytics Storage using Global Configuration
    market_basket_path = Path(cfg.get("data.paths.market_basket_export"))
    market_basket_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Cast back to Polars for lightning-fast CSV export
    pl.DataFrame(strong_rules).write_csv(str(market_basket_path))
    
    logger.info("Market Basket rules generated successfully!", rules_extracted=len(strong_rules), path=str(market_basket_path))

if __name__ == "__main__":
    compute_market_basket()
