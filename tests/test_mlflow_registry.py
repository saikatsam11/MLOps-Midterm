"""MLflow logging helpers and registry promotion, against a throwaway SQLite
tracking store (no server needed)."""
import json

import mlflow
import pytest
from mlflow import MlflowClient

import promote
from log_mlflow import flatten, run_metrics, run_params

MODEL = "test-model"

BIAS = {
    "overall_auc": 0.90,
    "final_score": 0.86,
    "power_means": {"subgroup_auc": 0.8, "bpsn_auc": 0.78, "bnsp_auc": 0.9},
    "n_eval": 100,
    "subgroups": {"muslim": {"n": 600, "subgroup_auc": 0.8, "bpsn_auc": 0.77, "bnsp_auc": 0.92}},
    "excluded_subgroups": {},
}
OVERALL = {
    "metrics_at_threshold_0_5": {"roc_auc": 0.93, "average_precision": 0.67, "f1": 0.60},
    "metrics_at_best_f1_threshold": {"f1": 0.61, "precision": 0.62, "recall": 0.60},
    "best_f1_threshold": 0.64,
}


def test_flatten_nested_params():
    assert flatten({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


def test_identity_columns_logged_as_count():
    params = run_params({"data": {"identity_columns": ["x", "y", "z"], "max_rows": 10}})
    assert params == {"data.identity_columns": 3, "data.max_rows": 10}


def test_run_metrics_include_every_subgroup_metric():
    metrics = run_metrics(OVERALL, BIAS)
    assert metrics["bias_final_score"] == 0.86
    assert metrics["valid_roc_auc"] == 0.93
    assert metrics["muslim.bpsn_auc"] == 0.77
    assert {"muslim.subgroup_auc", "muslim.bnsp_auc", "bias_power_mean_bpsn_auc"} <= set(metrics)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test")
    client = MlflowClient()
    client.create_registered_model(MODEL)

    def new_version(bias):
        bias_file = tmp_path / "bias.json"
        bias_file.write_text(json.dumps(bias))
        with mlflow.start_run() as run:
            mlflow.log_artifact(str(bias_file), artifact_path="metrics")
        mv = client.create_model_version(MODEL, f"runs:/{run.info.run_id}/model", run.info.run_id)
        run_info = tmp_path / f"run_v{mv.version}.json"
        run_info.write_text(json.dumps({"model_version": mv.version}))
        return mv.version, str(run_info)

    def report(promote_flag, failed=()):
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"promote": promote_flag, "failed": list(failed)}))
        return str(path)

    return tmp_path, client, new_version, report


def test_fetch_baseline_without_production_writes_nothing(registry):
    tmp_path, _, _, _ = registry
    out = tmp_path / "prod_bias.json"
    assert promote.main(["fetch-baseline", "--model-name", MODEL, "--out", str(out)]) == 0
    assert not out.exists()


def test_promote_then_fetch_baseline(registry):
    tmp_path, client, new_version, report = registry
    version, run_info = new_version(BIAS)
    args = ["--model-name", MODEL, "--run-info", run_info, "--report", report(True)]
    assert promote.main(["promote", *args]) == 0
    assert client.get_model_version_by_alias(MODEL, "production").version == version

    out = tmp_path / "prod_bias.json"
    assert promote.main(["fetch-baseline", "--model-name", MODEL, "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == BIAS


def test_promote_refuses_blocked_candidate(registry):
    _, client, new_version, report = registry
    _, run_info = new_version(BIAS)
    blocked = report(False, ["muslim.bpsn_auc"])
    assert promote.main(["promote", "--model-name", MODEL, "--run-info", run_info,
                         "--report", blocked]) == 1
    with pytest.raises(mlflow.exceptions.MlflowException):
        client.get_model_version_by_alias(MODEL, "production")


def test_reject_tags_version_and_keeps_production(registry):
    _, client, new_version, report = registry
    v1, run1 = new_version(BIAS)
    promote.main(["promote", "--model-name", MODEL, "--run-info", run1, "--report", report(True)])

    v2, run2 = new_version(BIAS)
    blocked = report(False, ["muslim.bpsn_auc"])
    assert promote.main(["reject", "--model-name", MODEL, "--run-info", run2,
                         "--report", blocked]) == 0
    tags = client.get_model_version(MODEL, v2).tags
    assert tags["gate"] == "blocked" and tags["gate_failed"] == "muslim.bpsn_auc"
    assert client.get_model_version_by_alias(MODEL, "production").version == v1
