import pandas as pd
import pytest
from src.data import load_messages
from src.model import create_pipeline


def test_dataset_labels(tmp_path):
    path = tmp_path / "spam.csv"
    pd.DataFrame({"Category": ["ham", "spam"], "Message": ["hello", "claim prize"]}).to_csv(path, index=False)
    df = load_messages(path)
    assert df["label"].tolist() == [0, 1]


def test_invalid_columns(tmp_path):
    path = tmp_path / "spam.csv"
    pd.DataFrame({"v1": ["ham"], "v2": ["hello"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Expected columns"):
        load_messages(path)


def test_pipeline_predicts():
    pipeline = create_pipeline()
    pipeline.fit(["hello friend", "good morning", "win money prize", "claim free prize"], [0, 0, 1, 1])
    assert len(pipeline.predict(["hello", "free prize"])) == 2
    assert pipeline.predict_proba(["hello"]).shape == (1, 2)
