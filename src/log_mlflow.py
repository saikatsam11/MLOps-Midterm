"""Log one pipeline run to MLflow and register the model.

Run after `dvc repro`. Reads params.yaml, metrics/metrics.json,
metrics/bias.json and the trained artifacts, logs them as a single MLflow run,
registers the model as a new version of the registered model and points the
`staging` alias at it. The run id and version are written to
metrics/mlflow_run.json for scripts/promote.py.

Tracking server: $MLFLOW_TRACKING_URI (default http://127.0.0.1:5000).
"""
from pathlib import Path
from importlib.metadata import version
import argparse
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from toxicity_pyfunc import ARTIFACT_FILES, ToxicityModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"


def flatten(d, prefix=""):
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def run_params(params):
    data = dict(params["data"])
    # 24 column names would be noise as a param; the count is what varies.
    data["identity_columns"] = len(data["identity_columns"])
    return flatten({**params, "data": data})


def run_metrics(overall, bias):
    metrics = {
        "valid_roc_auc": overall["metrics_at_threshold_0_5"]["roc_auc"],
        "valid_average_precision": overall["metrics_at_threshold_0_5"]["average_precision"],
        "valid_f1_at_0_5": overall["metrics_at_threshold_0_5"]["f1"],
        "valid_f1_best": overall["metrics_at_best_f1_threshold"]["f1"],
        "valid_precision_best": overall["metrics_at_best_f1_threshold"]["precision"],
        "valid_recall_best": overall["metrics_at_best_f1_threshold"]["recall"],
        "best_f1_threshold": overall["best_f1_threshold"],
        "bias_overall_auc": bias["overall_auc"],
        "bias_final_score": bias["final_score"],
        "bias_eligible_subgroups": len(bias["subgroups"]),
    }
    for metric, value in bias["power_means"].items():
        metrics[f"bias_power_mean_{metric}"] = value
    for group, row in bias["subgroups"].items():
        for metric in ("subgroup_auc", "bpsn_auc", "bnsp_auc"):
            metrics[f"{group}.{metric}"] = row[metric]
    return metrics


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", default="baseline",
                   help="Label for this candidate, e.g. baseline, imbalanced, reweighted")
    p.add_argument("--experiment", default="toxicity-bias")
    p.add_argument("--model-name", default="toxicity-classifier")
    p.add_argument("--out", default="metrics/mlflow_run.json")
    p.add_argument("--metrics-dir", default="metrics",
                   help="Folder holding metrics.json and bias.json for this run")
    a = p.parse_args()

    # MLflow prints emoji when a run ends; a Windows cp1252 console cannot
    # encode them and the script would crash after logging.
    sys.stdout.reconfigure(errors="replace")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(a.experiment)

    metrics_dir = ROOT / a.metrics_dir
    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    overall = json.loads((metrics_dir / "metrics.json").read_text(encoding="utf-8"))
    bias = json.loads((metrics_dir / "bias.json").read_text(encoding="utf-8"))
    data_md5 = yaml.safe_load((ROOT / "train.csv.dvc").read_text(encoding="utf-8"))["outs"][0]["md5"]
    artifact_dir = ROOT / "artifacts"

    tags = {
        "candidate": a.run_name,
        "git_commit": git_commit(),
        "data_md5": data_md5,
    }
    if os.environ.get("BUILD_TAG"):
        tags["jenkins_build"] = os.environ["BUILD_TAG"]

    with mlflow.start_run(run_name=a.run_name, tags=tags) as run:
        mlflow.log_params(run_params(params))
        mlflow.log_metrics(run_metrics(overall, bias))

        # Threshold sweep as stepped metrics so the UI draws P/R/F1 curves;
        # step is the threshold in hundredths.
        for row in overall["threshold_search"]:
            step = int(round(row["threshold"] * 100))
            for metric in ("precision", "recall", "f1"):
                mlflow.log_metric(f"sweep_{metric}", row[metric], step=step)

        for path in (metrics_dir / "metrics.json", metrics_dir / "bias.json", ROOT / "params.yaml"):
            mlflow.log_artifact(str(path), artifact_path="metrics")

        mlflow.log_artifact(str(artifact_dir / "model_metadata.json"), artifact_path="metrics")

        example = pd.DataFrame({"comment_text": ["have a nice day", "you are an idiot"]})
        # Output is one toxicity probability per input row.
        signature = infer_signature(example, np.array([0.1, 0.9]))

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=ToxicityModel(),
            artifacts={key: str(artifact_dir / name) for key, name in ARTIFACT_FILES.items()},
            code_paths=[str(ROOT / "src" / "toxicity_pyfunc.py")],
            signature=signature,
            input_example=example,
            pip_requirements=[
                f"{pkg}=={version(pkg)}"
                for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "mlflow")
            ],
            registered_model_name=a.model_name,
        )

    client = MlflowClient()
    model_version = str(info.registered_model_version)
    client.set_registered_model_alias(a.model_name, "staging", model_version)
    for key, value in {"candidate": a.run_name, "final_score": f"{bias['final_score']:.4f}"}.items():
        client.set_model_version_tag(a.model_name, model_version, key, value)

    out = ROOT / a.out
    out.write_text(json.dumps({
        "run_id": run.info.run_id,
        "experiment": a.experiment,
        "model_name": a.model_name,
        "model_version": model_version,
        "candidate": a.run_name,
    }, indent=2), encoding="utf-8")

    print(f"Logged run {run.info.run_id} ({a.run_name}) to {mlflow.get_tracking_uri()}")
    print(f"Registered {a.model_name} version {model_version} -> alias 'staging'")


if __name__ == "__main__":
    main()
