import pandas as pd

from data_processing import load_csv, filter_sbd, prepare_model_df, infer_feature_types


def make_sample_df():
    rows = [
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Raw",
            "BodyweightKg": "83.0",
            "AgeClass": "25-34",
            "TotalKg": "600",
            "Age": "28",
        },
        {
            "Event": "SBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": "63.5",
            "AgeClass": "25-34",
            "TotalKg": "350",
            "Age": "30",
        },
        {
            "Event": "SBD",
            "Sex": "M",
            "Equipment": "Equipped",
            "BodyweightKg": "105.0",
            "AgeClass": "35-44",
            "TotalKg": "700",
            "Age": "38",
        },
        {
            "Event": "NotSBD",
            "Sex": "F",
            "Equipment": "Raw",
            "BodyweightKg": "58.0",
            "AgeClass": "18-24",
            "TotalKg": "320",
            "Age": "22",
        },
    ]
    return pd.DataFrame(rows)


def test_load_csv_and_filter(tmp_path):
    df = make_sample_df()
    p = tmp_path / "sample.csv"
    df.to_csv(p, index=False)

    loaded = load_csv(p)
    assert isinstance(loaded, pd.DataFrame)
    assert "Event" in loaded.columns

    sbd = filter_sbd(loaded)
    assert all(sbd["Event"] == "SBD")
    assert sbd["TotalKg"].notna().all()


def test_prepare_and_infer():
    df = make_sample_df()
    features = ["Sex", "Equipment", "BodyweightKg", "AgeClass"]
    model_df = prepare_model_df(df, features, "TotalKg")
    # BodyweightKg should be coerced to numeric
    assert pd.api.types.is_numeric_dtype(model_df["BodyweightKg"])

    num, cat = infer_feature_types(model_df, features)
    assert "BodyweightKg" in num
    assert "Sex" in cat
