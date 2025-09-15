

import unittest
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Suppress LightGBM 'no further splits with positive gain' warnings
warnings.filterwarnings("ignore", message="No further splits with positive gain, best gain:*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

class TestPowerlifting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a small sample for testing
        cls.pd_df = pd.read_csv('openpowerlifting.csv', nrows=100)
        # Let Polars infer types and skip problematic rows
        cls.pl_df = pl.read_csv('openpowerlifting.csv', ignore_errors=True).head(100)

    def test_empty_dataframe(self):
        # Edge case: empty DataFrame
        empty_pd = pd.DataFrame(columns=self.pd_df.columns)
        empty_pl = pl.DataFrame(schema=self.pl_df.schema)
        self.assertTrue(empty_pd.empty)
        self.assertTrue(empty_pl.is_empty())
        # Filtering on empty should return empty
        sbd_pd = empty_pd[(empty_pd['Event'] == 'SBD') if 'Event' in empty_pd else []]
        self.assertTrue(sbd_pd.empty)
        # ML pipeline should fail gracefully
        features = ['Sex', 'Equipment', 'BodyweightKg', 'AgeClass']
        target = 'TotalKg'
        with self.assertRaises(Exception):
            X = empty_pd[features]
            y = empty_pd[target]
            model = Pipeline([
                ('preprocessor', ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Equipment', 'AgeClass']),
                    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), ['BodyweightKg'])
                ])),
                ('regressor', LGBMRegressor(objective='regression', random_state=253))
            ])
            model.fit(X, y)

    def test_missing_columns(self):
        # Edge case: missing required columns
        df = self.pd_df.drop(columns=['Sex']) if 'Sex' in self.pd_df else self.pd_df.copy()
        features = ['Sex', 'Equipment', 'BodyweightKg', 'AgeClass']
        with self.assertRaises(Exception):
            _ = df[features]

    def test_all_missing_values(self):
        # Edge case: all values missing in a feature
        df = self.pd_df.copy()
        df['BodyweightKg'] = None
        features = ['Sex', 'Equipment', 'BodyweightKg', 'AgeClass']
        target = 'TotalKg'
        sbd = df[(df['Event'] == 'SBD') & (df['TotalKg'].notna())]
        if len(sbd) < 10:
            self.skipTest('Not enough SBD rows for all-missing test')
        X = sbd[features]
        y = sbd[target]
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Equipment', 'AgeClass']),
            ('num', numeric_transformer, ['BodyweightKg'])
        ])
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(objective='regression', random_state=253))
        ])
        # Should raise ValueError, as all values are missing in BodyweightKg
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_no_sbd_events(self):
        # Edge case: no SBD events present
        df = self.pd_df[self.pd_df['Event'] != 'SBD']
        sbd = df[(df['Event'] == 'SBD') & (df['TotalKg'].notna())]
        self.assertTrue(sbd.empty)

    def test_invalid_values(self):
        # Edge case: negative weights
        df = self.pd_df.copy()
        df.loc[df.index[:5], 'BodyweightKg'] = -100
        self.assertTrue((df['BodyweightKg'] < 0).any())
        # ML pipeline should still run (LightGBM can handle negatives)
        features = ['Sex', 'Equipment', 'BodyweightKg', 'AgeClass']
        target = 'TotalKg'
        sbd = df[(df['Event'] == 'SBD') & (df['TotalKg'].notna())]
        if len(sbd) < 10:
            self.skipTest('Not enough SBD rows for invalid value test')
        X = sbd[features]
        y = sbd[target]
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Equipment', 'AgeClass']),
            ('num', numeric_transformer, ['BodyweightKg'])
        ])
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(objective='regression', random_state=253))
        ])
        try:
            model.fit(X, y)
        except Exception as e:
            self.fail(f"Model failed on negative values: {e}")

    def test_data_loading(self):
        # Test pandas and polars data loading
        self.assertFalse(self.pd_df.empty)
        self.assertFalse(self.pl_df.is_empty())
        self.assertIn('Event', self.pd_df.columns)
        self.assertIn('Event', self.pl_df.columns)

    def test_filtering(self):
        # Test SBD event filtering
        sbd_pd = self.pd_df[(self.pd_df['Event'] == 'SBD') & (self.pd_df['TotalKg'].notna())]
        sbd_pl = self.pl_df.filter((pl.col('Event') == 'SBD') & (pl.col('TotalKg').is_not_null()))
        self.assertTrue(len(sbd_pd) > 0)
        self.assertTrue(sbd_pl.height > 0)

    def test_ml_pipeline(self):
        # Test ML pipeline runs on small data
        features = ['Sex', 'Equipment', 'BodyweightKg', 'AgeClass']
        target = 'TotalKg'
        sbd_pl = self.pl_df.filter((pl.col('Event') == 'SBD') & (pl.col('TotalKg').is_not_null()))
        sbd_pd = sbd_pl.select(features + [target]).to_pandas()
        if len(sbd_pd) < 10:
            self.skipTest('Not enough SBD rows for ML test')
        X = sbd_pd[features]
        y = sbd_pd[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=253)
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Equipment', 'AgeClass']),
            ('num', numeric_transformer, ['BodyweightKg'])
        ])
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LGBMRegressor(objective='regression', random_state=253))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.assertTrue(mse >= 0)

if __name__ == '__main__':
    unittest.main()

