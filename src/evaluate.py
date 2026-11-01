from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import yaml
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]


def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype("int8")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }


def threshold_sweep(y_true, probabilities, eval_cfg):
    """Precision/recall/F1 over the threshold grid in params.yaml; returns the
    rows and the threshold with the best F1."""
    thresholds = np.arange(
        float(eval_cfg["threshold_min"]),
        float(eval_cfg["threshold_max"]) + 1e-9,
        float(eval_cfg["threshold_step"]),
    )
    rows = [
        {
            "threshold": float(t),
            "precision": float(precision_score(y_true, probabilities >= t, zero_division=0)),
            "recall": float(recall_score(y_true, probabilities >= t, zero_division=0)),
            "f1": float(f1_score(y_true, probabilities >= t, zero_division=0)),
        }
        for t in thresholds
    ]
    best = max(rows, key=lambda row: row["f1"])
    return rows, float(best["threshold"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(ROOT / "metrics"))
    a = p.parse_args()

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    data_cfg = params["data"]
    eval_cfg = params["evaluation"]
    processed = ROOT / "data" / "processed"
    artifact_dir = ROOT / "artifacts"
    metrics_dir = Path(a.out_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    valid_df = pd.read_csv(processed / "valid.csv")
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]
    x_valid = valid_df[text_col].fillna("").astype(str)
    y_valid = valid_df[label_col].astype("int8").to_numpy()

    word_vectorizer = joblib.load(artifact_dir / "word_tfidf_vectorizer.joblib")
    char_vectorizer = joblib.load(artifact_dir / "char_tfidf_vectorizer.joblib")
    model = joblib.load(artifact_dir / "toxicity_logistic_regression.joblib")

    print("Transforming validation comments...")
    x_word = word_vectorizer.transform(x_valid)
    x_char = char_vectorizer.transform(x_valid)
    x_valid_vec = hstack([x_word, x_char], format="csr", dtype=np.float32)
    probabilities = model.predict_proba(x_valid_vec)[:, list(model.classes_).index(1)]

    default_threshold = float(eval_cfg["default_threshold"])
    metrics_at_default = calculate_metrics(y_valid, probabilities, default_threshold)

    threshold_rows, best_threshold = threshold_sweep(y_valid, probabilities, eval_cfg)
    metrics_at_best = calculate_metrics(y_valid, probabilities, best_threshold)

    output = {
        "metrics_at_threshold_0_5": metrics_at_default,
        "best_f1_threshold": best_threshold,
        "metrics_at_best_f1_threshold": metrics_at_best,
        "threshold_search": threshold_rows,
        "validation_rows": int(len(y_valid)),
    }
    (metrics_dir / "metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    # Evaluation owns model_metadata.json so DVC stages have non-overlapping outputs.
    word_cfg = params["features"]["word"]
    char_cfg = params["features"]["char"]
    model_cfg = params["model"]
    metadata = {
        "model_name": "civil_comments_tfidf_logistic_regression",
        "dataset": "Jigsaw Civil Comments",
        "task": "binary toxicity classification",
        "text_column": text_col,
        "source_score_column": data_cfg["score_column"],
        "target_column": label_col,
        "target_rule": f"target >= {data_cfg['target_threshold']}",
        "random_state": int(data_cfg["random_state"]),
        "max_rows": data_cfg["max_rows"],
        "validation_size": float(data_cfg["validation_size"]),
        "default_threshold": default_threshold,
        "best_threshold": best_threshold,
        "validation_best_f1_threshold": best_threshold,
        "validation_metrics_at_0_5": {
            k: v for k, v in metrics_at_default.items() if k != "threshold"
        },
        "validation_metrics_at_best_f1_threshold": {
            k: v for k, v in metrics_at_best.items() if k != "threshold"
        },
        "word_vectorizer": {
            "analyzer": "word",
            "ngram_range": [int(word_cfg["ngram_min"]), int(word_cfg["ngram_max"])],
            "min_df": int(word_cfg["min_df"]),
            "max_features": int(word_cfg["max_features"]),
            "sublinear_tf": bool(word_cfg["sublinear_tf"]),
        },
        "char_vectorizer": {
            "analyzer": "char",
            "ngram_range": [int(char_cfg["ngram_min"]), int(char_cfg["ngram_max"])],
            "min_df": int(char_cfg["min_df"]),
            "max_features": int(char_cfg["max_features"]),
            "sublinear_tf": bool(char_cfg["sublinear_tf"]),
        },
        "classifier": {
            "type": "LogisticRegression",
            "C": float(model_cfg["C"]),
            "class_weight": model_cfg["class_weight"],
            "solver": model_cfg["solver"],
            "max_iter": int(model_cfg["max_iter"]),
        },
    }
    metadata_path = artifact_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(output, indent=2))
    print(f"Best validation F1 threshold: {best_threshold:.2f}")


if __name__ == "__main__":
    main()
