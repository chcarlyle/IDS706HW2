# Powerlifting Data Analysis

![CI](https://github.com/chcarlyle/IDS706HW2/actions/workflows/python-tests.yml/badge.svg)

This repository contains a small end-to-end pipeline that loads the OpenPowerlifting dataset, prepares features, fits several regression models to predict a competitor's 3-lift total (`TotalKg`), and produces artifacts (trained pipeline pickles and actual-vs-predicted plots).

Key points
- Tests are written with `pytest` and live under `tests/`.
- The main orchestration script is `powerlifting.py` which runs a roster of regressors, saves model artifacts into `outputs/`, and writes a `outputs/metrics.csv` summary.
- CI is configured with GitHub Actions and runs `flake8` and `pytest`.

---

## Automated testing

The repository includes a `tests/` directory with pytest-based unit tests that cover:

- Basic functionality: data loading, SBD filtering, model pipeline execution on a small sample.
- Edge cases: missing columns, empty dataframes, and other consistency checks.

Run the tests with:

```bash
pytest -q
```

All tests should pass in a correctly-configured Python environment.

## Dataset
The dataset originates from the OpenPowerlifting project (commonly distributed via Kaggle). The full CSV is large; a small sample `data/openpowerlifting_sample.csv` is provided for quick development and tests.

### Git LFS
If you store the full CSV with Git LFS, install and pull LFS files locally:

```bash
git lfs install
git lfs pull
```

Note: the GitHub Actions checkout step enables LFS if needed.

## What the scripts do

- `data_processing.py`: helpers to load CSV (pandas/polars fallback), filter for SBD, and prepare a modeling DataFrame.
- `models.py`: `train_and_evaluate_model(...)` builds a preprocessing + estimator pipeline, fits it, evaluates MSE, and optionally saves an artifact (joblib .pkl) containing `model`, `mse`, `X_test`, `y_test`, `y_pred`.
- `visualization.py`: `plot_actual_vs_predicted_from_pickle(...)` loads a saved artifact and writes a scatter plot of actual vs predicted values.
- `powerlifting.py`: orchestrates running a roster of regressors, saving artifacts to `outputs/` and writing `outputs/metrics.csv`.

## Visualization and outputs

Running `python powerlifting.py` will populate `outputs/` with artifacts and PNGs:

- `outputs/<model_name>_artifact.pkl` — saved joblib artifact with pipeline + predictions
- `outputs/<model_name>_actual_vs_predicted.png` — actual vs predicted plot
- `outputs/metrics.csv` — summary of model MSEs

## Findings
The current regressors examined show higher performance from LightGBM over the other candidates, Random Forest, Gradient Boosting, Ridge Regression, and Elastic-Net Regression. The models are evaluated using MSE, and the script is abstracted so other regressors can be tested, or other subsets of columns. LightBGM is also trained with tuned hyperparameters after it is chosen to achieve even lower MSE after selecting hyperparameters from Randomized Grid Search.

## Assignment 5 Components
Best predictive model:
![lgb_best](outputs/lgb_random_search_actual_vs_predicted.png)

Screenshots showing refactoring and successful CI runs
![Refactoring F2](outputs/Screenshot%202025-09-30%20110241.png)
![Refactoring Extract](outputs/Screenshot%202025-09-30%20110353.png)
![CI Passing](outputs/Screenshot%202025-09-30%20110924.png)

Flake8 and black were also implemented for the files.

For improving the project, I spent time revising the current structure so the code is stored in separate files. This allows me to call the functions like training models and generating output graphs with different sets of variables, hyperparameters, and even regressors since scikit-learn models are built in similar ways. Rather than arbitrarily using LightGBM for the model, this repository explores other model options, particularly other regressors, in the main script `powerlifting.py` to determine which one achieves the lowest error for this particular data set.