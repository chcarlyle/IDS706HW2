"""Lightweight visualization helpers for the powerlifting project.

Contains a helper to load a saved model/prediction artifact (joblib .pkl)
and plot actual vs predicted values.
"""

from pathlib import Path
from typing import Optional

import joblib
from matplotlib import pyplot as plt


def plot_actual_vs_predicted_from_pickle(
    pickle_path: str, out_path: Optional[str] = None
) -> str:
    """Load the saved artifact and produce an actual vs predicted plot.

    Parameters
    ----------
    pickle_path: str
        Path to the joblib .pkl file produced by `models.train_and_evaluate_model(..., save_path=...)`.
    out_path: Optional[str]
        Where to save the PNG. If None, saves to 'outputs/actual_vs_predicted_totalkg.png'.

    Returns
    -------
    str
        The path to the saved PNG file.
    """
    data = joblib.load(pickle_path)

    y_test = data.get("y_test")
    y_pred = data.get("y_pred")

    if y_test is None or y_pred is None:
        raise ValueError("Pickle must contain y_test and y_pred")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        out_path = out_dir / "actual_vs_predicted_totalkg.png"
    else:
        out_path = Path(out_path)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual TotalKg")
    plt.ylabel("Predicted TotalKg")
    plt.title("Actual vs Predicted TotalKg")
    plt.savefig(out_path)
    plt.close()

    return str(out_path)
