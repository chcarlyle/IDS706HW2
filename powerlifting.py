#Plotting and data manipulation libraries
import pandas as pd
from matplotlib import pyplot as plt
import polars as pl
#ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
#Load data
pf_df = pd.read_csv('data/openpowerlifting.csv')
pf_df_polars = pl.read_csv("data/openpowerlifting.csv", 
                           dtypes={"Age": pl.Float64})
#Inspect data
print(pf_df.head())
print(pf_df.describe())
print(pf_df.info())
print(pf_df.shape)
print(pf_df.isna().sum())

#Count rows of each event
print(pf_df['Event'].value_counts())
#Filter to just SBD (data on more than 1 or 2 lifts)
sbd_df = pf_df[(pf_df['Event'] == 'SBD') & (pf_df['TotalKg'].notna())]

#Inspect with polars
print(pf_df_polars.head())           # first rows
print(pf_df_polars.describe())       # summary stats (numeric cols)
print(pf_df_polars.schema)           # column names & types
print(pf_df_polars.shape)            # (rows, cols)
print(pf_df_polars.null_count()) 

#Filter polars
sbd_df_polars = pf_df_polars.filter((pl.col('Event') == 'SBD') & (pl.col('TotalKg').is_not_null()))

#Summary statistics for SBD lifting totals by sex and equipment
sbd_summary = sbd_df.groupby(['Sex', 'Equipment'])['TotalKg'].describe()
print(sbd_summary)

#Polars summary
sbd_summary_polars = (
    sbd_df_polars.group_by(["Sex", "Equipment"])
          .agg([
              pl.col("TotalKg").count().alias("count"),
              pl.col("TotalKg").mean().alias("mean"),
              pl.col("TotalKg").std().alias("std"),
              pl.col("TotalKg").min().alias("min"),
              pl.col("TotalKg").quantile(0.25).alias("q25"),
              pl.col("TotalKg").median().alias("median"),
              pl.col("TotalKg").quantile(0.75).alias("q75"),
              pl.col("TotalKg").max().alias("max"),
          ])
          .sort(["Sex", "Equipment"])
)
print(sbd_summary_polars)

#Define features and target
def default_model(sbd_df_polars, numeric_features, cate_features, target):
    features = numeric_features + cate_features
    target = target

#Define features in polars and convert to pandas for sklearn
    sbd_df = sbd_df_polars.select(features + [target]).to_pandas()

    X = sbd_df[features]
    y = sbd_df[target]

#Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=253)
#Preprocessing pipeline
#Impute median for numeric features
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])
#One-hot encode categorical features and apply numeric transformer
    preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cate_features),
        ('num', numeric_transformer, numeric_features + target)
    ]
)
#Create the model pipeline using lightgbm
    model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(objective='regression', random_state=253))
])

#Train default model
    model.fit(X_train, y_train)
#Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Default Mean Squared Error: {mse}')
    return y,X_train,X_test,y_train,y_test,model

y, X_train, X_test, y_train, y_test, model = default_model(sbd_df_polars)

n_estimators = [100, 200, 300]
max_depth = [3, 5, 7]
learning_rate = [0.01, 0.1, 0.2]
subsample = [0.6, 0.8, 1.0]
reg_alpha = [0, 0.1, 1]
reg_lambda = [1, 1.5, 2]

#Hyperparameter tuning with RandomizedSearchCV
def tune_hypers(X_train, X_test, y_train, y_test, model, n_estimators, max_depth, learning_rate, subsample, reg_alpha, reg_lambda):
    param_dist = {
    'regressor__n_estimators': n_estimators,
    'regressor__max_depth': max_depth,
    'regressor__learning_rate': learning_rate,
    'regressor__subsample': subsample,
    'regressor__reg_alpha': reg_alpha,
    'regressor__reg_lambda': reg_lambda
    }
    grid_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, scoring='neg_mean_squared_error', cv=3, verbose=1, random_state=253)
    grid_search.fit(X_train, y_train)
    print(f'Best parameters: {grid_search.best_params_}')

#Evaluate the best model
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    mse_best = mean_squared_error(y_test, y_pred_best)
    print(f'Best Model Mean Squared Error: {mse_best}')
#Save best model to outputs/
    joblib.dump(best_model, 'outputs/best_powerlifting_model.pkl')
    return y_pred_best

y_pred_best = tune_hypers(X_train, X_test, y_train, y_test, model, n_estimators, max_depth, learning_rate, subsample, reg_alpha, reg_lambda)

#Plot actual vs predicted
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual TotalKg')
plt.ylabel('Predicted TotalKg')
plt.title('Actual vs Predicted TotalKg')
plt.show()
#Save figure to outputs/
plt.savefig('outputs/actual_vs_predicted_totalkg.png')