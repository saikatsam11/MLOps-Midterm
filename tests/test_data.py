
import pandas as pd


def test_training_data_exists():
    df = pd.read_csv("data/processed/train.csv")

    assert not df.empty


def test_training_data_has_required_columns():
    df = pd.read_csv("data/processed/train.csv")

    assert "comment_text" in df.columns
    assert "toxic" in df.columns


def test_target_contains_valid_labels():
    df = pd.read_csv("data/processed/train.csv")

    assert set(df["toxic"].dropna().unique()).issubset({0, 1})
