import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge 
import xgboost as xgb 
import lightgbm as lgb 
import catboost as cb
import structlog
from typing import List, Dict, Any, Optional

logger = structlog.get_logger()

# For linear model's that need scaling for numerical features
def get_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Build a preprocessing ColumnTransformer for models lacking native categorical support.
    
    Applies simple median imputation and robust scaling for numeric data.
    Uses most frequent imputation and one-hot encoding for categorical data.

    Args:
        numeric_features (List[str]): List of numerical column names.
        categorical_features (List[str]): List of categorical column names.

    Returns:
        ColumnTransformer: An initialized scikit-learn ColumnTransformer ready for pipeline integration.
    """
    logger.debug("Building Preprocessor ColumnTransformer", num_num_features=len(numeric_features), num_cat_features=len(categorical_features))
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features), 
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor


def get_model_pipeline(
    model_type: str, 
    numeric_features: List[str], 
    categorical_features: List[str], 
    model_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Construct an end-to-end machine learning pipeline given a model type.

    The pipeline automatically applies a target transformation (np.log1p) during training
    and inverse transformation (np.expm1) during prediction to handle right-skewed sales.
    For advanced trees (XGBoost, LightGBM), it bypasses manual scaling/encoding.

    Args:
        model_type (str): The regression algorithm (e.g., 'XGBoost', 'LightGBM', 'RandomForest', 'LinearRegression').
        numeric_features (List[str]): List of numerical predictor column names.
        categorical_features (List[str]): List of categorical predictor column names.
        model_params (Optional[Dict[str, Any]]): Hyperparameters for the initialized model.

    Returns:
        Pipeline: Assembled, untrained scikit-learn Pipeline.

    Raises:
        ValueError: If the provided model_type is not supported.
    """
    if model_params is None:
        model_params = {}

    if model_type == 'XGBoost':
        regressor = xgb.XGBRegressor(**model_params)

    elif model_type == 'LightGBM':
        regressor = lgb.LGBMRegressor(verbose=-1, **model_params) 

    elif model_type == 'CatBoost':
        regressor = cb.CatBoostRegressor(verbose=-1, **model_params) 
    
    elif model_type == 'RandomForest':
        regressor = RandomForestRegressor(**model_params) 

    elif model_type == 'LinearRegression':
        regressor = Ridge(**model_params) 

    else:
        logger.error("Failed to parse model type!", requested_model=model_type)
        raise ValueError(f"Unknown model_type: {model_type}") 
    
    logger.info("Initializing Target Transformer Wrapper", algorithm=model_type) 

    wrapped_model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )

    if model_type in ['XGBoost', 'LightGBM']:
        logger.info("Constructing pure Booster Pipeline (Bypassing ColumnTransformer)")
        return Pipeline(steps=[('model', wrapped_model)])
    
    else:
        logger.info("Constructing Legacy Pipeline (Applying RobustScaler and OHE)")
        preprocessor = get_preprocessor(numeric_features, categorical_features)
        return Pipeline(steps=[('preprocessor', preprocessor), ('model', wrapped_model)])