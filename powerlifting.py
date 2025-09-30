"""Main script for the powerlifting modeling pipeline.

This script uses helpers in `data_processing.py` and `models.py` to load and
prepare the data, train a baseline model, perform randomized hyperparameter
search, save the best model, and produce an actual-vs-predicted plot.
"""

from pathlib import Path
import pandas as pd

from data_processing import load_csv, filter_sbd, prepare_model_df, infer_feature_types
from models import train_and_evaluate_model
from visualization import plot_actual_vs_predicted_from_pickle
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def main() -> None:
    data_path = Path("data/openpowerlifting.csv")
    df = load_csv(data_path)

    print(df.head())
    print(df.shape)
    print(df.isna().sum())

    sbd = filter_sbd(df)

    features = ["Sex", "Equipment", "BodyweightKg", "AgeClass"]
    target = "TotalKg"

    model_df = prepare_model_df(sbd, features, target)

    numeric_features, categorical_features = infer_feature_types(model_df, features)
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # Define model roster to try. We keep the RandomizedSearchCV LGBM as an
    # optional tuned estimator but also run a handful of common regressors for
    # quick comparison. Results (MSE) will be collected into outputs/metrics.csv.
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 1.5, 2],
    }

    # Small roster of estimators to compare
    roster = [
        ("lgb_baseline", LGBMRegressor(random_state=253)),
        ("random_forest", RandomForestRegressor(random_state=253)),
        ("gradient_boosting", GradientBoostingRegressor(random_state=253)),
        ("ridge", Ridge(random_state=253)),
        ("elasticnet", ElasticNet(random_state=253)),
    ]

    # We also include a tuned LGBM via RandomizedSearchCV as an extra spec
    roster.append(
        (
            "lgb_random_search",
            RandomizedSearchCV(
                LGBMRegressor(random_state=253),
                param_distributions=param_dist,
                n_iter=20,
                scoring="neg_mean_squared_error",
                cv=3,
                verbose=0,
                random_state=253,
            ),
        )
    )

    specs = []
    for name, estimator in roster:
        specs.append(
            {
                "name": name,
                "estimator": estimator,
                "save": f"outputs/{name}_artifact.pkl",
            }
        )

    Path("outputs").mkdir(parents=True, exist_ok=True)

    results = []

    for spec in specs:
        name = spec["name"]
        estimator = spec["estimator"]
        save_path = spec["save"]
        print("\nRunning model:", name)

        result = train_and_evaluate_model(
            df=model_df,
            features=features,
            target=target,
            estimator=estimator,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            save_path=save_path,
        )

        print(f"{name} MSE: {result['mse']}")

        results.append(
            {"name": name, "mse": float(result["mse"]), "artifact": save_path}
        )

        # Create a plot from the saved artifact (if created)
        try:
            png_out = f"outputs/{name}_actual_vs_predicted.png"
            plot_actual_vs_predicted_from_pickle(save_path, out_path=png_out)
            print("Saved plot to", png_out)
        except Exception as exc:  # pragma: no cover - plotting/io
            print("Failed to create plot for", name, exc)

    # Save a small CSV ranking the tried estimators by MSE
    try:
        metrics = pd.DataFrame(results).sort_values("mse")
        metrics.to_csv("outputs/metrics.csv", index=False)
        print("Saved metrics to outputs/metrics.csv")
    except Exception:
        pass


if __name__ == "__main__":
    main()
