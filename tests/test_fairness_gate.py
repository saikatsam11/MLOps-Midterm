import copy
import json

import pytest

from fairness_gate import evaluate_gate, main

TOL = {
    "overall_auc_tolerance": 0.002,
    "final_score_tolerance": 0.0,
    "subgroup_auc_tolerance": 0.01,
    "bpsn_auc_tolerance": 0.01,
}

PROD = {
    "overall_auc": 0.90,
    "final_score": 0.86,
    "subgroups": {
        "muslim": {"n": 5000, "subgroup_auc": 0.80, "bpsn_auc": 0.78, "bnsp_auc": 0.92},
        "white": {"n": 6000, "subgroup_auc": 0.80, "bpsn_auc": 0.73, "bnsp_auc": 0.94},
    },
}


def candidate(**changes):
    """Copy of PROD with dotted-path overrides, e.g. **{"muslim.bpsn_auc": 0.7}."""
    cand = copy.deepcopy(PROD)
    for path, value in changes.items():
        if "." in path:
            group, metric = path.split(".")
            cand["subgroups"][group][metric] = value
        else:
            cand[path] = value
    return cand


def failed(checks):
    return [c["check"] for c in checks if not c["passed"]]


def test_identical_model_is_promoted():
    promote, checks = evaluate_gate(candidate(), PROD, TOL)
    assert promote and failed(checks) == []


def test_no_production_baseline_promotes():
    assert evaluate_gate(candidate(), None, TOL) == (True, [])


def test_more_accurate_model_with_bpsn_regression_is_blocked():
    # The headline case: overall AUC and final score up, one subgroup's BPSN down.
    cand = candidate(overall_auc=0.93, final_score=0.88, **{"muslim.bpsn_auc": 0.74})
    promote, checks = evaluate_gate(cand, PROD, TOL)
    assert not promote
    assert failed(checks) == ["muslim.bpsn_auc"]


def test_regression_within_tolerance_passes():
    cand = candidate(**{"muslim.bpsn_auc": 0.77, "white.subgroup_auc": 0.79})
    promote, _ = evaluate_gate(cand, PROD, TOL)
    assert promote


def test_subgroup_auc_regression_is_blocked():
    promote, checks = evaluate_gate(candidate(**{"white.subgroup_auc": 0.78}), PROD, TOL)
    assert not promote and failed(checks) == ["white.subgroup_auc"]


def test_overall_auc_regression_is_blocked():
    promote, checks = evaluate_gate(candidate(overall_auc=0.897), PROD, TOL)
    assert not promote and failed(checks) == ["overall_auc"]


def test_final_score_regression_is_blocked():
    promote, checks = evaluate_gate(candidate(final_score=0.859), PROD, TOL)
    assert not promote and failed(checks) == ["final_score"]


def test_bnsp_is_not_gated():
    promote, _ = evaluate_gate(candidate(**{"white.bnsp_auc": 0.5}), PROD, TOL)
    assert promote


def test_missing_subgroup_is_blocked():
    cand = candidate()
    del cand["subgroups"]["white"]
    promote, checks = evaluate_gate(cand, PROD, TOL)
    assert not promote and failed(checks) == ["white.present"]


@pytest.fixture
def files(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text("gate:\n" + "".join(f"  {k}: {v}\n" for k, v in TOL.items()))

    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(payload))
        return str(path)

    return tmp_path, str(params), write


def test_cli_exit_codes_and_report(files):
    tmp_path, params, write = files
    prod = write("prod.json", PROD)
    report = tmp_path / "report.json"

    good = write("good.json", candidate())
    assert main(["--candidate", good, "--production", prod,
                 "--params", params, "--report", str(report)]) == 0
    assert json.loads(report.read_text())["promote"] is True

    bad = write("bad.json", candidate(**{"muslim.bpsn_auc": 0.70}))
    assert main(["--candidate", bad, "--production", prod,
                 "--params", params, "--report", str(report)]) == 1
    assert json.loads(report.read_text())["failed"] == ["muslim.bpsn_auc"]


def test_cli_missing_production_file_promotes(files):
    tmp_path, params, write = files
    cand = write("cand.json", candidate())
    report = tmp_path / "report.json"
    assert main(["--candidate", cand, "--production", str(tmp_path / "none.json"),
                 "--params", params, "--report", str(report)]) == 0
    assert json.loads(report.read_text())["has_production_baseline"] is False
