# Model training and bias evaluation

How to reproduce the baseline model and its bias metrics from scratch.

## 1. Environment

Python 3.12. From the repository root:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements-train.txt
```

## 2. Data

The dataset is versioned with DVC (`train.csv.dvc`, 816,211,476 bytes,
MD5 `7a687c299462dd009e90db97ac053eef`). Fetch it into the repository root:

```bash
dvc pull
```

> Content warning: the dataset contains abusive and offensive text. It is never
> committed to git; `*.csv` and `data/` are git-ignored.

## 3. Run the pipeline

```bash
dvc repro
```

Stages (`dvc.yaml`):

| Stage | Script | Output |
|---|---|---|
| `prepare` | `src/prepare.py` | `data/processed/train.csv`, `valid.csv`, `identity_eval.parquet` |
| `train` | `src/train.py` | TF-IDF vectorizers + logistic regression in `artifacts/` |
| `evaluate` | `src/evaluate.py` | `metrics/metrics.json` (overall AUC, F1, threshold sweep) |
| `bias_eval` | `src/bias_eval.py` | `metrics/bias.json` (per-identity bias metrics) |

All settings live in `params.yaml`. `dvc repro` reruns only the stages whose
code, data or parameters changed. View metrics with `dvc metrics show`.

### Identity hold-out

Identity labels exist only on a subset of comments. `prepare.py` first samples
`data.identity_eval_size` identity-labelled rows (fixed seed) into
`identity_eval.parquet` and removes them from the pool *before* the training
sample is drawn, so bias metrics are always computed on rows the model never
saw and are comparable across runs.

## 4. Bias metrics

From Borkan et al., *Nuanced Metrics for Measuring Unintended Bias with Real
Data for Text Classification* (arXiv:1903.04561). A comment belongs to an
identity subgroup when its identity score is `>= 0.5`.

| Metric | Computed on | Low value means |
|---|---|---|
| Subgroup AUC | comments mentioning the identity | model cannot rank toxicity within the subgroup |
| BPSN AUC | non-toxic subgroup + toxic background | non-toxic identity mentions are over-flagged (false positives) |
| BNSP AUC | toxic subgroup + non-toxic background | toxic identity comments are missed (false negatives) |

Final score:

```
score = 0.25 * overall_auc + 0.25 * M(subgroup_auc) + 0.25 * M(bpsn_auc) + 0.25 * M(bnsp_auc)
M(x)  = ( mean(x_s ^ p) ) ^ (1/p),  p = -5
```

The negative power makes the worst subgroup dominate. Subgroups with fewer
than `bias.min_subgroup_size` (500) evaluation rows are listed under
`excluded_subgroups` and left out of the score, since their AUCs are noisy.

`metrics/bias.json` schema (read by the fairness gate):

```json
{
  "overall_auc": 0.0,
  "final_score": 0.0,
  "power_means": {"subgroup_auc": 0.0, "bpsn_auc": 0.0, "bnsp_auc": 0.0},
  "n_eval": 0,
  "subgroups": {
    "muslim": {"n": 0, "subgroup_auc": 0.0, "bpsn_auc": 0.0, "bnsp_auc": 0.0}
  },
  "excluded_subgroups": {"buddhist": 0}
}
```

## 5. Fairness gate

`scripts/fairness_gate.py` decides whether a newly trained model may replace
the production model. It compares the candidate's `bias.json` with the
production model's `bias.json`:

```
PROMOTE if
    overall_auc                >= prod - 0.002
    final_score                >= prod
    subgroup_auc, every group  >= prod - 0.01
    bpsn_auc,     every group  >= prod - 0.01
BLOCK otherwise
```

Tolerances live in the `gate` section of `params.yaml`. A subgroup that is in
production but missing from the candidate also blocks. BNSP is reported in
`bias.json` but not gated.

```bash
# first model: no production baseline yet, always promoted
python scripts/fairness_gate.py --candidate metrics/bias.json

# compare against production
python scripts/fairness_gate.py --candidate metrics/bias.json --production prod_bias.json
```

Exit code `0` = promote, `1` = blocked, so a CI stage fails on a block. Every
check is printed as a table and saved to `metrics/gate_report.json` before
the decision. A blocked run ends with the offending checks named, e.g.:

```
muslim.bpsn_auc      0.7451   0.7751  -0.0300  -0.0100  FAIL
FAIRNESS GATE BLOCKED: 1 check(s) regressed: muslim.bpsn_auc
```

## 6. Tests

```bash
pytest -q tests
```

- `tests/test_bias_metrics.py` checks every metric against values counted by
  hand on a small golden fixture.
- `tests/test_fairness_gate.py` covers promotion, blocking on each check
  (including a more accurate model that regresses one subgroup), tolerance
  edges, a missing subgroup, no production baseline, and the CLI exit codes.
