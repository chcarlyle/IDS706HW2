import pandas as pd
from sklearn.linear_model import Ridge

from models import train_and_evaluate_model


def make_small_df():
    rows = [
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": 83.0,
            "AgeClass": "25-34",
            "TotalKg": 600.0,
        },
        {
            "Event": "SBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": 63.5,
            "AgeClass": "25-34",
            "TotalKg": 350.0,
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Equipped",
            "BodyweightKg": 105.0,
            "AgeClass": "35-44",
            "TotalKg": 700.0,
        },
        {
            "Event": "SBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": 58.0,
            "AgeClass": "18-24",
            "TotalKg": 320.0,
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": 90.0,
            "AgeClass": "25-34",
            "TotalKg": 650.0,
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": 85.0,
            "AgeClass": "25-34",
            "TotalKg": 620.0,
        },
    ]
    return pd.DataFrame(rows)


def test_train_and_evaluate(tmp_path):
    df = make_small_df()
    features = ["Sex", "Equipment", "BodyweightKg", "AgeClass"]
    target = "TotalKg"

    result = train_and_evaluate_model(
        df=df,
        features=features,
        target=target,
        estimator=Ridge(),
        test_size=0.3,
        random_state=42,
        save_path=str(tmp_path / "artifact.pkl"),
    )

    assert "model" in result and "mse" in result
    assert result["mse"] >= 0
    # saved artifact should exist
    assert (tmp_path / "artifact.pkl").exists()
