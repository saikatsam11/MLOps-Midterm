"""Fairness gate: decide whether a candidate model may replace production.

Compares the candidate's bias.json against the production model's bias.json.
The candidate is BLOCKED if overall AUC, the final bias-aware score, or any
subgroup's subgroup AUC / BPSN AUC regresses by more than its tolerance in
params.yaml, even when overall AUC improves.

Exits 0 to promote and 1 to block, so a CI stage fails on a block. A JSON
report of every check is written before the decision so it can be archived.
"""
from pathlib import Path
import argparse
import json
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]

# (metric in bias.json, tolerance key in params.yaml)
GLOBAL_CHECKS = [
    ("overall_auc", "overall_auc_tolerance"),
    ("final_score", "final_score_tolerance"),
]
SUBGROUP_CHECKS = [
    ("subgroup_auc", "subgroup_auc_tolerance"),
    ("bpsn_auc", "bpsn_auc_tolerance"),
]


def _check(name, candidate, production, tolerance):
    delta = candidate - production
    return {
        "check": name,
        "candidate": candidate,
        "production": production,
        "delta": delta,
        "tolerance": tolerance,
        # Small epsilon so a drop of exactly the tolerance is not failed by
        # floating-point rounding.
        "passed": delta >= -tolerance - 1e-9,
    }


def evaluate_gate(candidate, production, tolerances):
    """Return (promote, checks). production=None means no model is in
    production yet, so the candidate is promoted."""
    if production is None:
        return True, []

    checks = [
        _check(metric, candidate[metric], production[metric], tolerances[key])
        for metric, key in GLOBAL_CHECKS
    ]
    for name, prod_row in sorted(production["subgroups"].items()):
        cand_row = candidate["subgroups"].get(name)
        if cand_row is None:
            # Missing means its AUCs could not be measured, so a regression
            # cannot be ruled out.
            checks.append({"check": f"{name}.present", "passed": False})
            continue
        for metric, key in SUBGROUP_CHECKS:
            checks.append(
                _check(f"{name}.{metric}", cand_row[metric], prod_row[metric], tolerances[key])
            )
    return all(c["passed"] for c in checks), checks


def _print_report(promote, checks):
    print(f"{'check':<44}{'candidate':>10}{'prod':>9}{'delta':>9}{'allowed':>9}  result")
    for c in checks:
        if "delta" not in c:
            print(f"{c['check']:<44}{'missing':>10}{'':>9}{'':>9}{'':>9}  FAIL")
            continue
        print(
            f"{c['check']:<44}{c['candidate']:>10.4f}{c['production']:>9.4f}"
            f"{c['delta']:>+9.4f}{-c['tolerance']:>+9.4f}  {'ok' if c['passed'] else 'FAIL'}"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--candidate", default="metrics/bias.json")
    p.add_argument("--production", default=None,
                   help="Production bias.json; omit or point to a missing file "
                        "when nothing is in production yet")
    p.add_argument("--params", default=str(ROOT / "params.yaml"))
    p.add_argument("--report", default="metrics/gate_report.json")
    a = p.parse_args(argv)

    tolerances = yaml.safe_load(Path(a.params).read_text(encoding="utf-8"))["gate"]
    candidate = json.loads(Path(a.candidate).read_text(encoding="utf-8"))
    production = None
    if a.production and Path(a.production).is_file():
        production = json.loads(Path(a.production).read_text(encoding="utf-8"))

    promote, checks = evaluate_gate(candidate, production, tolerances)
    failed = [c["check"] for c in checks if not c["passed"]]

    report_path = Path(a.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {"promote": promote, "has_production_baseline": production is not None,
             "tolerances": tolerances, "failed": failed, "checks": checks},
            indent=2,
        ),
        encoding="utf-8",
    )

    if production is None:
        print("No production baseline found; first model is promoted.")
        print("FAIRNESS GATE PASSED.")
        return 0

    _print_report(promote, checks)
    if not promote:
        print(f"FAIRNESS GATE BLOCKED: {len(failed)} check(s) regressed: {', '.join(failed)}")
        return 1
    print("FAIRNESS GATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
