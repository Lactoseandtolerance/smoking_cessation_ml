from src.smoking_cessation_ml.data import split_features_target
import pandas as pd


def test_split_features_target():
    df = pd.DataFrame({"a": [1, 2], "quit": [0, 1]})
    X, y = split_features_target(df, target_col="quit")
    assert "quit" not in X.columns
    assert y.tolist() == [0, 1]
