from pathlib import Path
import json
import re
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


def mentions(texts, terms):
    """Boolean mask of texts that contain any of the terms as whole words."""
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b"
    return texts.str.contains(pattern, case=False, regex=True).to_numpy()


def apply_candidate(texts, y, cfg, seed):
    """Return (keep mask, sample weights) for the training strategy in `cfg`
    (the `candidate` section of params.yaml)."""
    keep = np.ones(len(y), dtype=bool)
    weights = np.ones(len(y), dtype=np.float64)
    strategy = cfg["strategy"]
    if strategy == "imbalanced":
        sub = cfg["imbalanced"]
        target = mentions(texts, sub["terms"]) & (y == 0)
        rng = np.random.default_rng(seed)
        keep &= ~(target & (rng.random(len(y)) < float(sub["drop_fraction"])))
    elif strategy == "reweighted":
        sub = cfg["reweighted"]
        weights[mentions(texts, sub["terms"]) & (y == 0)] = float(sub["weight"])
    elif strategy != "standard":
        raise ValueError(f"Unknown candidate strategy: {strategy}")
    return keep, weights


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
    y_train = train_df[label_col].astype("int8").to_numpy()

    keep, sample_weight = apply_candidate(
        x_train, y_train, params["candidate"], int(data_cfg["random_state"])
    )
    x_train, y_train, sample_weight = x_train[keep], y_train[keep], sample_weight[keep]
    print(
        f"Candidate strategy '{params['candidate']['strategy']}': "
        f"{len(y_train):,} training rows ({(~keep).sum():,} dropped), "
        f"{(sample_weight != 1).sum():,} up-weighted"
    )

    word_vectorizer = build_word_vectorizer(feat_cfg["word"])
    char_vectorizer = build_char_vectorizer(feat_cfg["char"])

    print("Fitting word TF-IDF...")
    x_word = word_vectorizer.fit_transform(x_train)
    print("Fitting character TF-IDF...")
    x_char = char_vectorizer.fit_transform(x_train)
    x_train_vec = hstack([x_word, x_char], format="csr", dtype=np.float32)
    print("Training feature matrix:", x_train_vec.shape)

    model = build_model(model_cfg, int(data_cfg["random_state"]))
    model.fit(x_train_vec, y_train, sample_weight=sample_weight)

    joblib.dump(word_vectorizer, artifact_dir / "word_tfidf_vectorizer.joblib")
    joblib.dump(char_vectorizer, artifact_dir / "char_tfidf_vectorizer.joblib")
    joblib.dump(model, artifact_dir / "toxicity_logistic_regression.joblib")

    print("Saved model artifacts to", artifact_dir)


if __name__ == "__main__":
    main()
