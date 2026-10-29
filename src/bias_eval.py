"""Bias metrics on the identity-labelled hold-out (Borkan et al., arXiv:1903.04561).

Writes metrics/bias.json, which the fairness gate reads. The metric functions
take plain arrays so any model's scores can be evaluated the same way.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import yaml
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def _auc(y_true, scores):
    # AUC is undefined when only one class is present.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def subgroup_auc(y_true, scores, in_subgroup):
    """AUC restricted to comments that mention the identity."""
    return _auc(y_true[in_subgroup], scores[in_subgroup])


def bpsn_auc(y_true, scores, in_subgroup):
    """Background Positive, Subgroup Negative: low values mean false positives
    on non-toxic comments that mention the identity."""
    mask = (in_subgroup & (y_true == 0)) | (~in_subgroup & (y_true == 1))
    return _auc(y_true[mask], scores[mask])


def bnsp_auc(y_true, scores, in_subgroup):
    """Background Negative, Subgroup Positive: low values mean false negatives
    on toxic comments that mention the identity."""
    mask = (in_subgroup & (y_true == 1)) | (~in_subgroup & (y_true == 0))
    return _auc(y_true[mask], scores[mask])


def power_mean(values, power):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values ** power) ** (1.0 / power))


def compute_bias_metrics(y_true, scores, identities, power, weights, min_subgroup_size):
    """Compute the bias.json payload.

    y_true: binary labels, scores: predicted probabilities,
    identities: {name: boolean membership array}.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)

    subgroups, excluded = {}, {}
    for name, members in identities.items():
        members = np.asarray(members, dtype=bool)
        n = int(members.sum())
        row = {
            "n": n,
            "subgroup_auc": subgroup_auc(y_true, scores, members),
            "bpsn_auc": bpsn_auc(y_true, scores, members),
            "bnsp_auc": bnsp_auc(y_true, scores, members),
        }
        aucs_defined = all(np.isfinite(row[k]) for k in ("subgroup_auc", "bpsn_auc", "bnsp_auc"))
        if n >= min_subgroup_size and aucs_defined:
            subgroups[name] = row
        else:
            excluded[name] = n

    if not subgroups:
        raise ValueError("No subgroup meets min_subgroup_size; cannot compute bias score.")

    overall = _auc(y_true, scores)
    means = {
        metric: power_mean([row[metric] for row in subgroups.values()], power)
        for metric in ("subgroup_auc", "bpsn_auc", "bnsp_auc")
    }
    final = (
        weights["overall"] * overall
        + weights["subgroup"] * means["subgroup_auc"]
        + weights["bpsn"] * means["bpsn_auc"]
        + weights["bnsp"] * means["bnsp_auc"]
    )
    return {
        "overall_auc": overall,
        "final_score": float(final),
        "power_means": means,
        "n_eval": int(len(y_true)),
        "subgroups": subgroups,
        "excluded_subgroups": excluded,
    }


def tfidf_scores(texts, artifact_dir):
    word_vectorizer = joblib.load(artifact_dir / "word_tfidf_vectorizer.joblib")
    char_vectorizer = joblib.load(artifact_dir / "char_tfidf_vectorizer.joblib")
    model = joblib.load(artifact_dir / "toxicity_logistic_regression.joblib")
    x_vec = hstack(
        [word_vectorizer.transform(texts), char_vectorizer.transform(texts)],
        format="csr",
        dtype=np.float32,
    )
    return model.predict_proba(x_vec)[:, list(model.classes_).index(1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default=None,
                   help="CSV with a `score` column, one row per identity_eval.parquet row "
                        "in the same order (e.g. DistilBERT predictions). Default: score "
                        "with the TF-IDF artifacts.")
    p.add_argument("--out-dir", default=str(ROOT / "metrics"))
    a = p.parse_args()

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    data_cfg = params["data"]
    bias_cfg = params["bias"]
    metrics_dir = Path(a.out_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    eval_df = pd.read_parquet(ROOT / "data" / "processed" / "identity_eval.parquet")
    texts = eval_df[data_cfg["text_column"]].fillna("").astype(str)
    y_true = eval_df[data_cfg["label_column"]].astype("int8").to_numpy()

    if a.scores:
        scores = pd.read_csv(a.scores)["score"].to_numpy(dtype=float)
        if len(scores) != len(eval_df):
            raise ValueError(f"{a.scores} has {len(scores)} scores, expected {len(eval_df)}")
    else:
        print(f"Scoring {len(eval_df):,} identity-labelled comments...")
        scores = tfidf_scores(texts, ROOT / "artifacts")

    threshold = float(data_cfg["identity_threshold"])
    identities = {
        col: (eval_df[col] >= threshold).to_numpy()
        for col in data_cfg["identity_columns"]
    }
    result = compute_bias_metrics(
        y_true,
        scores,
        identities,
        power=float(bias_cfg["power"]),
        weights=bias_cfg["weights"],
        min_subgroup_size=int(bias_cfg["min_subgroup_size"]),
    )
    (metrics_dir / "bias.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Overall AUC: {result['overall_auc']:.4f}")
    print(f"Final bias-aware score: {result['final_score']:.4f}")
    print(f"{'subgroup':<32}{'n':>7}{'sub_auc':>9}{'bpsn':>8}{'bnsp':>8}")
    for name, row in sorted(result["subgroups"].items(), key=lambda kv: kv[1]["bpsn_auc"]):
        print(
            f"{name:<32}{row['n']:>7}{row['subgroup_auc']:>9.4f}"
            f"{row['bpsn_auc']:>8.4f}{row['bnsp_auc']:>8.4f}"
        )
    if result["excluded_subgroups"]:
        print("Excluded (below min_subgroup_size):", result["excluded_subgroups"])


if __name__ == "__main__":
    main()
