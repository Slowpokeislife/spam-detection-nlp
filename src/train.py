"""Reproducible train/evaluate/save entry point."""
import argparse
import json
from pathlib import Path
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from .data import load_messages
from .model import create_pipeline


def train(dataset: str, output: str, report: str, seed: int = 42) -> dict:
    df = load_messages(dataset)
    # Stratified split avoids changing class ratios between sets.
    x_train, x_test, y_train, y_test = train_test_split(
        df["Message"], df["label"], test_size=0.2, random_state=seed,
        stratify=df["label"],
    )
    pipeline = create_pipeline()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    result = {
        "dataset_rows": len(df),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "random_state": seed,
        "class_mapping": {"ham": 0, "spam": 1},
        "classification_report": classification_report(
            y_test, predictions, target_names=["ham", "spam"],
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    Path(report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(classification_report(y_test, predictions, target_names=["ham", "spam"], zero_division=0))
    print(f"Saved model: {output} | Evaluation report: {report}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SMS spam detection baseline")
    parser.add_argument("--data", default="data/raw/spam.csv")
    parser.add_argument("--model", default="models/spam_pipeline.joblib")
    parser.add_argument("--report", default="reports/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.model, args.report, args.seed)


if __name__ == "__main__":
    main()
