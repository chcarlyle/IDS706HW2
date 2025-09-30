"""Modeling functions to be used

This file provides a small, explicit function that trains any sklearn-like
regressor (one that implements fit/predict) on a DataFrame using a provided
list of feature column names and a target column name. It handles simple
preprocessing (median imputation for numeric columns and one-hot encoding for
categorical columns) and returns the fitted pipeline and evaluation results.

"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train_and_evaluate_model(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    estimator: Any,
    test_size: float = 0.2,
    random_state: int = 253,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Train and evaluate a regression model on the given DataFrame.

    Parameters
    ----------
    df: pd.DataFrame
        The full dataset containing features and the target column.
    features: List[str]
        Column names to use as predictors. These must exist in `df`.
    target: str
        Name of the numeric target column in `df`.
    estimator: Any
        A scikit-learn-like regressor instance with `fit` and `predict`.
    test_size: float
        Fraction of the data to hold out for testing.
    random_state: int
        Random seed used for train/test split and estimator reproducibility.
    numeric_features: Optional[List[str]]
        If provided, explicitly treated as numeric (imputed). Otherwise inferred.
    categorical_features: Optional[List[str]]
        If provided, explicitly treated as categorical (one-hot encoded). Otherwise inferred.

    Returns
    -------
    result : dict
        A dictionary with keys:
        - 'model': the fitted Pipeline (preprocessor + estimator)
        - 'mse': mean squared error on the test set
        - 'X_test', 'y_test', 'y_pred' for inspection

    Notes
    -----
    This function builds a simple ColumnTransformer that imputes missing
    numeric values with the median and one-hot encodes categorical columns.
    It wraps the estimator in an sklearn Pipeline so the returned model can be
    used directly for prediction on raw DataFrame rows.
    """

    # Basic validation
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    missing = set(features) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns not found in dataframe: {missing}")

    # Infer numeric / categorical features if not provided
    X = df[features].copy()
    if numeric_features is None or categorical_features is None:
        inferred_numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        inferred_categorical = [c for c in features if c not in inferred_numeric]
        if numeric_features is None:
            numeric_features = inferred_numeric
        if categorical_features is None:
            categorical_features = inferred_categorical

    # Ensure lists are not None
    numeric_features = numeric_features or []
    categorical_features = categorical_features or []

    # Preprocessing: median imputer for numeric, one-hot for categoricals
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    # Use sparse_output for newer scikit-learn versions; older versions used
    try:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )

    preprocessor = ColumnTransformer(
        # Pipeline for all predictors
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Build pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", estimator)]
    )

    # Split
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    result = {
        "model": pipeline,
        "mse": mse,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }

    if save_path:
        # Save pipeline and test split/predictions for visualization or later use
        try:
            import joblib

            joblib.dump(result, save_path)
        except Exception:
            pass

    return result
