"""Model training and persistence utilities."""
from typing import Any
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_pipeline() -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipe


def train_model(X, y) -> Any:
    pipe = build_pipeline()
    pipe.fit(X, y)
    return pipe


def save_model(model: Any, path: str):
    joblib.dump(model, path)


def load_model(path: str) -> Any:
    return joblib.load(path)
