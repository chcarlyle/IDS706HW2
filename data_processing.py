"""Data processing helper functions for the powerlifting project.

These functions prepare raw CSV data so it can be passed to the modeling functions
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def load_csv(
    path: Union[str, Path],
    use_polars: bool = False,
    polars_dtypes: Optional[Dict] = None,
) -> pd.DataFrame:
    """Load a CSV file and return a pandas DataFrame.

    Parameters
    ----------
    path: str | Path
        Path to CSV file.
    use_polars: bool
        If True, read with polars then convert to pandas. Useful for large files.
    polars_dtypes: Optional[dict]
        When using polars, optional dtypes mapping to pass to pl.read_csv.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if use_polars:
        try:
            import polars as pl

            if polars_dtypes is None:
                df_pl = pl.read_csv(str(path))
            else:
                df_pl = pl.read_csv(str(path), dtypes=polars_dtypes)
            return df_pl.to_pandas()
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Failed to read CSV with polars") from exc
    else:
        return pd.read_csv(path)


def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise ValueError if any required columns are missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_sbd(df: pd.DataFrame) -> pd.DataFrame:
    """Return only SBD event rows with a non-null TotalKg.

    Keeps existing columns; does not drop rows with missing features. The
    modeling helper will handle imputation.
    """
    if "Event" not in df.columns or "TotalKg" not in df.columns:
        raise ValueError("DataFrame must contain 'Event' and 'TotalKg' columns")
    return df[(df["Event"] == "SBD") & (df["TotalKg"].notna())].copy()


def prepare_model_df(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    dropna_on_target: bool = True,
    coerce_numeric: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame containing only the requested features and target.

    Parameters
    ----------
    df: pd.DataFrame
        Source dataframe.
    features: list[str]
        Feature column names to select.
    target: str
        Target column name.
    dropna_on_target: bool
        If True, drop rows where target is missing.
    coerce_numeric: bool
        If True, attempt to convert numeric-looking columns to numeric dtype.

    Returns
    -------
    pd.DataFrame
        A copy of the selected columns ready for modeling (imputation still
        required for missing feature values).
    """
    validate_required_columns(df, [target] + features)
    model_df = df[features + [target]].copy()

    if dropna_on_target:
        model_df = model_df[model_df[target].notna()].copy()

    if coerce_numeric:
        for col in features + [target]:
            # try to convert to numeric where it makes sense
            if model_df[col].dtype == object:
                coerced = pd.to_numeric(model_df[col], errors="coerce")
                # If conversion produced non-nulls where original was non-null,
                # keep the numeric version for that column
                if coerced.notna().sum() >= (model_df[col].notna().sum() / 2):
                    model_df[col] = coerced

    return model_df


# Separate feature types for pre-processing pipeline
def infer_feature_types(
    df: pd.DataFrame, features: List[str]
) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical features from a DataFrame.

    Returns (numeric_features, categorical_features).
    """
    numeric = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical
