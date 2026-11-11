"""Read and validate the Kaggle SMS spam dataset."""
from pathlib import Path
import pandas as pd


def load_messages(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Download spam.csv from "
            "https://www.kaggle.com/datasets/tmehul/spamcsv and place it in data/raw/."
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Original notebook expects the columns Category and Message.
    required = {"Category", "Message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(required)}; found {list(df.columns)}")
    df = df[["Category", "Message"]].dropna().copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    if not df["Category"].isin(["ham", "spam"]).all():
        raise ValueError("Category must contain only 'ham' and 'spam'.")
    df["Message"] = df["Message"].astype(str)
    df["label"] = (df["Category"] == "spam").astype(int)
    return df.reset_index(drop=True)
