"""Data loading and preprocessing helpers."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Keeps it minimal: checks for file existence errors are left to the caller.
    """
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str = "quit") -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in dataframe")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test(df: pd.DataFrame, target_col: str = "quit", test_size: float = 0.2, random_state: int = 42):
    X, y = split_features_target(df, target_col)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
