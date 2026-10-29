"""Model registry steps around the fairness gate.

  fetch-baseline  download the bias.json of the model aliased `production`
                  (writes nothing if no model is in production yet)
  promote         move the `production` alias to the candidate, only if the
                  gate report says promote
  reject          tag the candidate as blocked by the gate

The candidate is read from metrics/mlflow_run.json (written by
src/log_mlflow.py). Tracking server: $MLFLOW_TRACKING_URI
(default http://127.0.0.1:5000).
"""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import tempfile
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
PRODUCTION, STAGING = "production", "staging"


def production_version(client, model_name):
    try:
        return client.get_model_version_by_alias(model_name, PRODUCTION)
    except MlflowException:
        # Raised when the registered model or the alias does not exist yet.
        return None


def fetch_baseline(client, a):
    out = Path(a.out)
    if out.exists():
        out.unlink()
    prod = production_version(client, a.model_name)
    if prod is None:
        print(f"No '{PRODUCTION}' version of {a.model_name}; gate will treat this as the first model.")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            run_id=prod.run_id, artifact_path="metrics/bias.json", dst_path=tmp
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, out)
    print(f"Production baseline: {a.model_name} v{prod.version} (run {prod.run_id}) -> {out}")
    return 0


def promote(client, a, run_info, report):
    version = run_info["model_version"]
    if not report["promote"]:
        print(f"Refusing to promote v{version}: gate report says blocked ({', '.join(report['failed'])}).")
        return 1
    previous = production_version(client, a.model_name)
    client.set_registered_model_alias(a.model_name, PRODUCTION, version)
    client.set_model_version_tag(a.model_name, version, "gate", "passed")
    if previous is not None and previous.version != version:
        client.set_model_version_tag(a.model_name, previous.version, "replaced_by", version)
    was = f"v{previous.version}" if previous else "none"
    print(f"Promoted {a.model_name} v{version} to '{PRODUCTION}' (was {was}).")
    return 0


def reject(client, a, run_info, report):
    version = run_info["model_version"]
    client.set_model_version_tag(a.model_name, version, "gate", "blocked")
    client.set_model_version_tag(a.model_name, version, "gate_failed", ", ".join(report["failed"]))
    try:
        if client.get_model_version_by_alias(a.model_name, STAGING).version == version:
            client.delete_registered_model_alias(a.model_name, STAGING)
    except MlflowException:
        pass
    print(f"Marked {a.model_name} v{version} as blocked by the fairness gate.")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="Model registry steps around the fairness gate.")
    p.add_argument("action", choices=["fetch-baseline", "promote", "reject"])
    p.add_argument("--model-name", default="toxicity-classifier")
    p.add_argument("--out", default="metrics/prod_bias.json",
                   help="fetch-baseline: where to write the production bias.json")
    p.add_argument("--run-info", default="metrics/mlflow_run.json")
    p.add_argument("--report", default="metrics/gate_report.json")
    a = p.parse_args(argv)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    client = MlflowClient()

    if a.action == "fetch-baseline":
        return fetch_baseline(client, a)

    run_info = json.loads(Path(a.run_info).read_text(encoding="utf-8"))
    report = json.loads(Path(a.report).read_text(encoding="utf-8"))
    action = promote if a.action == "promote" else reject
    return action(client, a, run_info, report)


if __name__ == "__main__":
    sys.exit(main())
