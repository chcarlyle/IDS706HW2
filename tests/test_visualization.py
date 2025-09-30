import joblib
import pandas as pd
from pathlib import Path

from visualization import plot_actual_vs_predicted_from_pickle


def test_plot_from_pickle(tmp_path):
    # Create synthetic artifact and save
    y_test = pd.Series([100.0, 150.0, 200.0])
    y_pred = pd.Series([110.0, 140.0, 195.0])
    artifact = {"y_test": y_test, "y_pred": y_pred}
    p = tmp_path / "artifact.pkl"
    joblib.dump(artifact, p)

    out_png = tmp_path / "out.png"
    saved = plot_actual_vs_predicted_from_pickle(str(p), out_path=str(out_png))
    assert Path(saved).exists()
