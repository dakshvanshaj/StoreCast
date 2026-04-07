import great_expectations as gx 
import polars as pl 
import pandas as pd
import structlog 
import config 
import shutil 
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

def rules_sales(suite):
    """
    Expectation rules for sales dataset.
    Parameters:
        suite: GX ExpectationSuite
    """
    # Base check: Columns match exactly
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=["store", "dept", "date", "weekly_sales", "isholiday"], exact_match=True
    ))

    # Existing Rules
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="weekly_sales", min_value=0.0))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="store", min_value=1, max_value=45))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="date"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="weekly_sales"))

    # Primary Key Uniqueness
    suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["store", "dept", "date"]))
    
    # Type checks
    float_types = ["float", "float32", "float64"]
    int_types = ["int", "int32", "int64"]
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column="weekly_sales", type_list=float_types))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column="store", type_list=int_types))

def rules_features(suite):
    """
    Expectation rules for features dataset.
    Parameters:
        suite: GX ExpectationSuite
    """
    # Base check: Columns match exactly
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=["store", "date", "temperature", "fuel_price", "markdown1", "markdown2", "markdown3", "markdown4", "markdown5", "cpi", "unemployment", "isholiday"], exact_match=True
    ))

    # Nulls & Bounding
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="cpi"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="unemployment"))

    # Primary Key Uniqueness
    suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(column_list=["store", "date"]))
    
    float_types = ["float", "float32", "float64"]
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column="cpi", type_list=float_types))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column="unemployment", type_list=float_types))

    for i in range(1, 6):
        col = f"markdown{i}"
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column=col, min_value=0.0))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column=col, type_list=float_types))

def rules_stores(suite):
    """
    Expectation rules for stores dataset.
    Parameters:
        suite: GX ExpectationSuite
    """
    # Base check: Columns match exactly
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=["store", "type", "size"], exact_match=True
    ))

    # Types, Bounding & Categorical Enums
    int_types = ["int", "int32", "int64"]
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="store", min_value=1, max_value=45))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInTypeList(column="store", type_list=int_types))
    
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeUnique(column="store"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="type", value_set=["A", "B", "C"]))

def validate():
    """
    Main validation function.
    """
    logger.info('Great Expectations Validation Started')

    logger.info('Reading Silver Data', dataset='sales')
    sales_df = pl.read_delta(str(config.SILVER_SALES_PATH)).to_pandas()
    logger.info('Reading Silver Data', dataset='features')
    features_df = pl.read_delta(str(config.SILVER_FEATURES_PATH)).to_pandas()
    logger.info('Reading Silver Data', dataset='stores')
    stores_df = pl.read_delta(str(config.SILVER_STORES_PATH)).to_pandas()

    logger.info('Initializing GX Context', mode='ephemeral')
    context = gx.get_context(mode='ephemeral')

    res_sales = build_and_run_validation(context, sales_df, "sales", rules_sales)
    res_features = build_and_run_validation(context, features_df, "features", rules_features)
    res_stores = build_and_run_validation(context, stores_df, "stores", rules_stores)

    # Summarize results
    all_success = res_sales.success and res_features.success and res_stores.success

    if all_success:
        logger.info('Great Expectations Validation PASSED!', layer='silver')
    else:
        logger.error('Great Expectations Validation FAILED!', 
                     sales_success=res_sales.success, 
                     features_success=res_features.success, 
                     stores_success=res_stores.success)

    
    # GENERATE HTML REPORT
    logger.info('Building Data Docs HTML Report...')
    docs_info = context.build_data_docs()
    
    # Copy from /tmp to a permanent local directory
    tmp_path_str = docs_info.get('local_site', '').replace("file://", "")
    
    if tmp_path_str:
        tmp_dir = Path(tmp_path_str).parent
        target_dir = config.PROJECT_ROOT / "docs" / "gx_data_docs_silver"
        
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