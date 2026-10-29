"""Grid search for the TF-IDF + logistic regression model, tracked in MLflow.

Each combination in `tuning.grid` (params.yaml) is logged as a nested child run
under one parent run, with its validation metrics and its bias metrics on the
identity hold-out. Trials train on a `tuning.train_rows` subsample of the
prepared train split.

Writes metrics/tuning/results.csv. Run after `dvc repro` (needs
data/processed/). Tracking server: $MLFLOW_TRACKING_URI
(default http://127.0.0.1:5000).
"""
from pathlib import Path
import argparse
import itertools
import os
import sys
import time
import numpy as np
import pandas as pd
import yaml
import mlflow
from scipy.sparse import hstack

from bias_eval import compute_bias_metrics
from evaluate import calculate_metrics, threshold_sweep
from log_mlflow import DEFAULT_TRACKING_URI, git_commit, run_metrics
from train import build_char_vectorizer, build_model, build_word_vectorizer

ROOT = Path(__file__).resolve().parents[1]


def positive_proba(model, x):
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="tfidf-grid-search")
    p.add_argument("--out-dir", default="metrics/tuning")
    a = p.parse_args()

    sys.stdout.reconfigure(errors="replace")  # see log_mlflow.py
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    data_cfg, feat_cfg, bias_cfg = params["data"], params["features"], params["bias"]
    tune_cfg = params["tuning"]
    grid = tune_cfg["grid"]
    seed = int(data_cfg["random_state"])
    text_col, label_col = data_cfg["text_column"], data_cfg["label_column"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(tune_cfg["experiment"])

    processed = ROOT / "data" / "processed"
    train_df = pd.read_csv(processed / "train.csv")
    train_df = train_df.sample(n=min(int(tune_cfg["train_rows"]), len(train_df)), random_state=seed)
    valid_df = pd.read_csv(processed / "valid.csv")
    eval_df = pd.read_parquet(processed / "identity_eval.parquet")
    texts = {name: df[text_col].fillna("").astype(str)
             for name, df in (("train", train_df), ("valid", valid_df), ("eval", eval_df))}
    y_train = train_df[label_col].astype("int8").to_numpy()
    y_valid = valid_df[label_col].astype("int8").to_numpy()
    y_eval = eval_df[label_col].astype("int8").to_numpy()
    identities = {col: (eval_df[col] >= float(data_cfg["identity_threshold"])).to_numpy()
                  for col in data_cfg["identity_columns"]}

    # Features do not depend on the grid, so they are built once for all trials.
    print(f"Fitting TF-IDF features on {len(train_df):,} training rows...")
    word_vec = build_word_vectorizer(feat_cfg["word"])
    char_vec = build_char_vectorizer(feat_cfg["char"])
    x = {"train": hstack([word_vec.fit_transform(texts["train"]),
                          char_vec.fit_transform(texts["train"])], format="csr", dtype=np.float32)}
    for k in ("valid", "eval"):
        x[k] = hstack([word_vec.transform(texts[k]), char_vec.transform(texts[k])],
                      format="csr", dtype=np.float32)

    combos = list(itertools.product(grid["C"], grid["class_weight"]))
    rows = []
    with mlflow.start_run(run_name=a.run_name, tags={"git_commit": git_commit(),
                                                     "candidate": "tuning"}) as parent:
        mlflow.log_params({"grid.C": str(grid["C"]),
                           "grid.class_weight": str(grid["class_weight"]),
                           "train_rows": len(train_df)})

        for i, (c, class_weight) in enumerate(combos, start=1):
            trial_params = {"C": c, "class_weight": class_weight or "none"}
            t0 = time.time()
            model = build_model({**params["model"], "C": c, "class_weight": class_weight}, seed)
            model.fit(x["train"], y_train)
            fit_seconds = time.time() - t0

            p_valid = positive_proba(model, x["valid"])
            _, best_t = threshold_sweep(y_valid, p_valid, params["evaluation"])
            overall = {
                "metrics_at_threshold_0_5": calculate_metrics(y_valid, p_valid, 0.5),
                "metrics_at_best_f1_threshold": calculate_metrics(y_valid, p_valid, best_t),
                "best_f1_threshold": best_t,
            }
            bias = compute_bias_metrics(
                y_eval, positive_proba(model, x["eval"]), identities,
                power=float(bias_cfg["power"]), weights=bias_cfg["weights"],
                min_subgroup_size=int(bias_cfg["min_subgroup_size"]),
            )
            metrics = run_metrics(overall, bias)
            metrics["min_bpsn_auc"] = min(r["bpsn_auc"] for r in bias["subgroups"].values())
            metrics["fit_seconds"] = fit_seconds

            name = f"C{c}-{trial_params['class_weight']}"
            with mlflow.start_run(run_name=name, nested=True, tags={"candidate": "tuning-trial"}):
                mlflow.log_params({**trial_params, "train_rows": len(train_df)})
                mlflow.log_metrics(metrics)

            rows.append({**trial_params, **{k: metrics[k] for k in (
                "valid_roc_auc", "valid_f1_best", "bias_final_score", "min_bpsn_auc")}})
            print(f"[{i}/{len(combos)}] {name}: valid AUC {metrics['valid_roc_auc']:.4f} | "
                  f"F1 {metrics['valid_f1_best']:.4f} | final score {metrics['bias_final_score']:.4f} | "
                  f"min BPSN {metrics['min_bpsn_auc']:.4f} | fit {fit_seconds:.0f}s")

        results = pd.DataFrame(rows).sort_values("valid_roc_auc", ascending=False)
        out_dir = ROOT / a.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "results.csv", index=False)
        mlflow.log_artifact(str(out_dir / "results.csv"))

    print(f"\nParent run {parent.info.run_id}, {len(combos)} trials:")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
