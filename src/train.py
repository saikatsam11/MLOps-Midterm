from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import yaml
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]


def build_word_vectorizer(word_cfg):
    return TfidfVectorizer(
        lowercase=word_cfg["lowercase"],
        strip_accents=word_cfg["strip_accents"],
        analyzer="word",
        ngram_range=(int(word_cfg["ngram_min"]), int(word_cfg["ngram_max"])),
        min_df=int(word_cfg["min_df"]),
        max_features=int(word_cfg["max_features"]),
        sublinear_tf=bool(word_cfg["sublinear_tf"]),
        dtype=np.float32,
    )


def build_char_vectorizer(char_cfg):
    return TfidfVectorizer(
        lowercase=char_cfg["lowercase"],
        analyzer="char",
        ngram_range=(int(char_cfg["ngram_min"]), int(char_cfg["ngram_max"])),
        min_df=int(char_cfg["min_df"]),
        max_features=int(char_cfg["max_features"]),
        sublinear_tf=bool(char_cfg["sublinear_tf"]),
        dtype=np.float32,
    )


def build_model(model_cfg, seed):
    return LogisticRegression(
        C=float(model_cfg["C"]),
        class_weight=model_cfg["class_weight"],
        solver=model_cfg["solver"],
        max_iter=int(model_cfg["max_iter"]),
        random_state=seed,
    )


def main():
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    data_cfg = params["data"]
    feat_cfg = params["features"]
    model_cfg = params["model"]
    processed = ROOT / "data" / "processed"
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed / "train.csv")
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]
    x_train = train_df[text_col].fillna("").astype(str)
    y_train = train_df[label_col].astype("int8")

    word_vectorizer = build_word_vectorizer(feat_cfg["word"])
    char_vectorizer = build_char_vectorizer(feat_cfg["char"])

    print("Fitting word TF-IDF...")
    x_word = word_vectorizer.fit_transform(x_train)
    print("Fitting character TF-IDF...")
    x_char = char_vectorizer.fit_transform(x_train)
    x_train_vec = hstack([x_word, x_char], format="csr", dtype=np.float32)
    print("Training feature matrix:", x_train_vec.shape)

    model = build_model(model_cfg, int(data_cfg["random_state"]))
    model.fit(x_train_vec, y_train)

    joblib.dump(word_vectorizer, artifact_dir / "word_tfidf_vectorizer.joblib")
    joblib.dump(char_vectorizer, artifact_dir / "char_tfidf_vectorizer.joblib")
    joblib.dump(model, artifact_dir / "toxicity_logistic_regression.joblib")

    print("Saved model artifacts to", artifact_dir)


if __name__ == "__main__":
    main()
