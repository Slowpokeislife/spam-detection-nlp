"""CLI inference from a saved, trusted model."""
import argparse
import joblib


def predict(texts: list[str], model_path: str = "models/spam_pipeline.joblib") -> list[dict]:
    if not texts or any(not item.strip() for item in texts):
        raise ValueError("Provide one or more non-empty messages.")
    # Only load joblib files created locally / from trusted sources.
    pipeline = joblib.load(model_path)
    labels = pipeline.predict(texts)
    scores = pipeline.predict_proba(texts)
    return [
        {"message": message, "label": "spam" if label == 1 else "ham",
         "spam_probability": round(float(prob[1]), 4)}
        for message, label, prob in zip(texts, labels, scores)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify SMS messages as spam or ham")
    parser.add_argument("messages", nargs="+", help="Messages in quotes")
    parser.add_argument("--model", default="models/spam_pipeline.joblib")
    args = parser.parse_args()
    for item in predict(args.messages, args.model):
        print(f"{item['label'].upper():4} | p(spam)={item['spam_probability']:.4f} | {item['message']}")


if __name__ == "__main__":
    main()
