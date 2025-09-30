import unittest
import pandas as pd
import polars as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Simple, robust tests: load data from data/ (real CSV or sample) or fallback to an in-memory sample.
warnings.filterwarnings("ignore")


def load_data(nrows=100):
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    real = data_dir / "openpowerlifting.csv"
    sample = data_dir / "openpowerlifting_sample.csv"

    if real.exists():
        pd_df = pd.read_csv(real, nrows=nrows)
        pl_df = pl.read_csv(str(real), ignore_errors=True).head(nrows)
        return pd_df, pl_df

    if sample.exists():
        pd_df = pd.read_csv(sample, nrows=nrows)
        pl_df = pl.read_csv(str(sample), ignore_errors=True).head(nrows)
        return pd_df, pl_df

    # in-memory fallback small dataset
    rows = [
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": 83.0,
            "AgeClass": "25-34",
            "TotalKg": 600.0,
            "Age": 28,
        },
        {
            "Event": "SBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": 63.5,
            "AgeClass": "25-34",
            "TotalKg": 350.0,
            "Age": 30,
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Equipped",
            "BodyweightKg": 105.0,
            "AgeClass": "35-44",
            "TotalKg": 700.0,
            "Age": 38,
        },
        {
            "Event": "SBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": 58.0,
            "AgeClass": "18-24",
            "TotalKg": 320.0,
            "Age": 22,
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": 90.0,
            "AgeClass": "25-34",
            "TotalKg": 650.0,
            "Age": 27,
        },
    ]
    df = pd.DataFrame(rows)
    return df, pl.from_pandas(df)


PD_DF, PL_DF = load_data()


class TestPowerlifting(unittest.TestCase):
    def test_data_loading(self):
        self.assertFalse(PD_DF.empty)
        self.assertFalse(PL_DF.is_empty())
        self.assertIn("Event", PD_DF.columns)

    def test_filtering(self):
        sbd_pd = PD_DF[(PD_DF["Event"] == "SBD") & (PD_DF["TotalKg"].notna())]
        sbd_pl = PL_DF.filter(
            (pl.col("Event") == "SBD") & (pl.col("TotalKg").is_not_null())
        )
        self.assertTrue(len(sbd_pd) > 0)
        self.assertTrue(sbd_pl.height > 0)

    def test_ml_pipeline_runs(self):
        features = ["Sex", "Equipment", "BodyweightKg", "AgeClass"]
        target = "TotalKg"
        sbd_pl = PL_DF.filter(
            (pl.col("Event") == "SBD") & (pl.col("TotalKg").is_not_null())
        )
        sbd_pd = sbd_pl.select(features + [target]).to_pandas()
        if len(sbd_pd) < 10:
            self.skipTest("Not enough rows for ML test")
        X = sbd_pd[features]
        y = sbd_pd[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Sex", "Equipment", "AgeClass"],
                ),
                ("num", numeric_transformer, ["BodyweightKg"]),
            ]
        )
        model = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", LGBMRegressor(objective="regression", random_state=42)),
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
