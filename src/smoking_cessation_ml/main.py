"""Small CLI entrypoint to train a model given a CSV."""
import argparse
from pathlib import Path

from .data import load_csv, train_test
from .models import train_model, save_model
from .utils import classification_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train a smoking cessation model")
    p.add_argument("--data", required=True, help="Path to CSV data file")
    p.add_argument("--target", default="quit", help="Target column name")
    p.add_argument("--out", default="models/model.joblib", help="Output model path")
    return p.parse_args()


def main():
    args = parse_args()
    df = load_csv(args.data)
    X_train, X_test, y_train, y_test = train_test(df, target_col=args.target)
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    metrics = classification_metrics(y_test, preds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_model(model, args.out)
    print("Trained model saved to", args.out)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
