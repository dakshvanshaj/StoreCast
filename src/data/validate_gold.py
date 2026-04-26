import great_expectations as gx 
import pandas as pd
import structlog 
from src.utils.config_manager import ConfigManager
import shutil 
import time
from pathlib import Path


logger = structlog.get_logger()

def build_and_run_validation(context, df, dataset_name, expectation_rules_func):
    """
    Helper function to build and run validation for a given dataset.
    Parameters:
        context: GX Context
        df: Pandas DataFrame
        dataset_name: Name of the dataset
        expectation_rules_func: Function to add expectations to the suite
    Returns:
        ValidationResult
    """
    logger.info(f"Adding Pandas Data Source", dataset=dataset_name)
    data_source = context.data_sources.add_pandas(f'silver_pandas_source_{dataset_name}')
    data_asset = data_source.add_dataframe_asset(f'{dataset_name}_asset') 
    batch_definition = data_asset.add_batch_definition_whole_dataframe(f'{dataset_name}_batch')

    logger.info(f"Creating Expectations Suite", dataset=dataset_name)
    suite = gx.ExpectationSuite(name=f'silver_suite_{dataset_name}')
    
    # Add rules specific to dataset
    expectation_rules_func(suite)
    context.suites.add(suite)

    logger.info(f'Creating Validation Definition', dataset=dataset_name)
    validation_def = gx.ValidationDefinition(
        data=batch_definition,
        suite=suite,
        name=f'silver_validation_{dataset_name}'
    )
    context.validation_definitions.add(validation_def)
    
    logger.info(f'Running Validation', dataset=dataset_name)
    return validation_def.run(batch_parameters={"dataframe": df})

def rules_master_sales(suite):
    """
    Expectation rules for sales dataset.
    Parameters:
        suite: GX ExpectationSuite
    """
    # 1. Base check: Columns match exactly
    expected_columns = [
        "store", "dept", "date", "weekly_sales", "isholiday",
        "store_type", "store_size", 
        "temperature", "fuel_price", "cpi", "unemployment",
        "markdown1", "markdown2", "markdown3", "markdown4", "markdown5",
        "total_markdown", "month", "week_of_year",
        "sales_log", "sin_week", "cos_week",
        "lag_1", "lag_5", "lag_52", "sales_last_year",
        "rolling_4_wk_log_sales_avg", "cpi_lag_3_month"
    ]
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=expected_columns, exact_match=True
    ))

    # 2. Target variable must NEVER be null and must be >= 0
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="weekly_sales"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="weekly_sales", min_value=0.0))

    # 3. Check if DuckDB successfully appended our ML features
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="sales_log"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="sin_week"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="lag_1"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="rolling_4_wk_log_sales_avg"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="cpi_lag_3_month"))

    # 4. Primary Keys uniquely identify a row (no Cartesian explosions)
    suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["store", "dept", "date"]))

    # 5. Check for data quality in store size (should be positive)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="store_size", min_value=0.0))

    # 6. Check for data quality in fuel price (should be positive)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="fuel_price", min_value=0.0))

    # 7. Check for data quality in CPI (should be positive)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="cpi", min_value=0.0))

    # 8. Check for data quality in unemployment (should be between 0 and 100)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="unemployment", min_value=0.0, max_value=100.0))

    # 9. Check for data quality in markdown columns (should be >= 0)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="markdown1", min_value=0.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="markdown2", min_value=0.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="markdown3", min_value=0.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="markdown4", min_value=0.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="markdown5", min_value=0.0))

    # 10. Check for data quality in total markdown (should be >= 0)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="total_markdown", min_value=0.0))

    # 11. Check for data quality in month (should be between 1 and 12)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="month", min_value=1, max_value=12))

    # 12. Check for data quality in week of year (should be between 1 and 53)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="week_of_year", min_value=1, max_value=53))

    # 13. Fourier Cyclical features must be between -1 and 1
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="sin_week", min_value=-1.0, max_value=1.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="cos_week", min_value=-1.0, max_value=1.0))

    # 14. Check for data quality in rolling average log sales (should be >= 0)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="rolling_4_wk_log_sales_avg", min_value=0.0))

    # 15. Check for data quality in sales last year (should be >= 0)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="sales_last_year", min_value=0.0))

    # 16. Check for data quality in store type (should be one of the expected values)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="store_type", value_set=["A", "B", "C"]))

    # 17. Check for data quality in isholiday (should be 0 or 1, and never NULL)
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="isholiday"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="isholiday", value_set=[0, 1]))

   
def validate():
    """
    Main validation function.
    """
    start_time = time.time()
    logger.info('Great Expectations Validation Started')

    cfg = ConfigManager()
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    logger.info('Reading Gold Data', dataset='master_sales')
    master_sales_df = pd.read_parquet(cfg.get("data.paths.gold_data"))
    
    logger.info('Initializing GX Context', mode='ephemeral')
    context = gx.get_context(mode='ephemeral')

    res_sales = build_and_run_validation(context, master_sales_df, "master_sales", rules_master_sales)
 
    # Summarize results
    all_success = res_sales.success
    duration = round(time.time() - start_time, 2)

    if all_success:
        logger.info('Great Expectations Validation PASSED!', layer='gold', duration_seconds=duration)
    else:
        logger.error('Great Expectations Validation FAILED!', 
                     master_sales_success=res_sales.success,
                     duration_seconds=duration)

    
    # GENERATE HTML REPORT
    logger.info('Building Data Docs HTML Report...')
    docs_info = context.build_data_docs()
    
    # Copy from /tmp to a permanent local directory
    tmp_path_str = docs_info.get('local_site', '').replace("file://", "")
    
    if tmp_path_str:
        tmp_dir = Path(tmp_path_str).parent
        target_dir = project_root / "docs" / "gx_data_docs_gold"
        
        if tmp_dir.exists():
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.copytree(tmp_dir, target_dir)
            logger.info('Data Docs safely saved to permanent directory', path=f'file://{target_dir}/index.html')
        else:
            logger.warning('Data Docs tmp directory not found.', path=tmp_path_str)
    else:
        logger.warning('Data Docs built without local_site path.', set=docs_info)

if __name__ == "__main__":
    validate()