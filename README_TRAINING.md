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

## 6. MLflow tracking and model registry

### Start the tracking server

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

UI: <http://127.0.0.1:5000>. Runs and the registry live in `mlflow.db`, model
files in `mlartifacts/` (both git-ignored). Scripts use
`$MLFLOW_TRACKING_URI`, defaulting to `http://127.0.0.1:5000`.

### Log, gate and promote a model

After `dvc repro`:

```bash
python src/log_mlflow.py --run-name baseline        # 1. log run, register new version, alias @staging
python scripts/promote.py fetch-baseline            # 2. download @production's bias.json (none on first run)
python scripts/fairness_gate.py --candidate metrics/bias.json --production metrics/prod_bias.json   # 3. gate
python scripts/promote.py promote                   # 4a. gate passed: move @production to the candidate
python scripts/promote.py reject                    # 4b. gate blocked: tag the candidate as blocked
```

| Logged to each run | |
|---|---|
| Params | every value in `params.yaml` (flattened, e.g. `model.C`, `bias.power`) |
| Metrics | validation AUC / F1 / precision / recall, best threshold, overall AUC and final score on the identity hold-out, power means, and `<identity>.subgroup_auc` / `.bpsn_auc` / `.bnsp_auc` for every eligible subgroup |
| Curves | `sweep_precision`, `sweep_recall`, `sweep_f1` over the threshold sweep (step = threshold x 100) |
| Tags | `candidate`, `git_commit`, `data_md5`, `model_family`, `jenkins_build` (when run in Jenkins) |
| Artifacts | `metrics/metrics.json`, `metrics/bias.json`, `params.yaml`, `model_metadata.json`, and the model |

### Registry

The model is registered as **`toxicity-classifier`**: an MLflow pyfunc that
bundles both TF-IDF vectorizers and the classifier (`src/toxicity_pyfunc.py`).
Lifecycle uses registry aliases (MLflow 3 replaces stages with aliases):

| Alias | Meaning |
|---|---|
| `@staging` | most recently logged candidate |
| `@production` | the model that passed the fairness gate and is serving |

Blocked candidates are tagged `gate=blocked` with the failed checks in
`gate_failed`; promoted ones are tagged `gate=passed`.

Load the production model anywhere:

```python
import mlflow, pandas as pd
model = mlflow.pyfunc.load_model("models:/toxicity-classifier@production")
model.predict(pd.DataFrame({"comment_text": ["I am a gay man."]}))
```

## 7. Jenkins CD pipeline

Jenkins and MLflow run together in Docker (`jenkins/`). Jenkins reaches the
tracking server at `http://mlflow:5000`.

### Start

Requires Docker Desktop (or Docker Engine) with Compose.

```bash
docker compose -f jenkins/docker-compose.yml up -d --build
```

| Service | URL | Container |
|---|---|---|
| Jenkins | <http://localhost:8080> | `jenkins-toxicity` |
| MLflow | <http://localhost:5000> | `mlflow-toxicity` |

If port 8080 is taken, choose another: `JENKINS_PORT=8081 docker compose -f jenkins/docker-compose.yml up -d --build`.

Stop with `docker compose -f jenkins/docker-compose.yml down` (add `-v` to
also delete Jenkins and MLflow data).

### First-time Jenkins setup

1. Get the initial admin password:
   `docker exec jenkins-toxicity cat /var/jenkins_home/secrets/initialAdminPassword`
2. Open Jenkins, paste it, choose **Install suggested plugins**, create an admin user.
3. **New Item** > name `toxicity-cd` > **Pipeline** > OK.
4. Under **Pipeline**: Definition **Pipeline script from SCM**, SCM **Git**,
   Repository URL `https://github.com/saikatsam11/MLOps-Midterm.git`,
   Branch `*/main`, Script Path `Jenkinsfile`. Save.
5. Data access for `DATA_SOURCE=dvc`: **Manage Jenkins > Credentials > Global >
   Add Credentials**, kind **Secret file**, ID `gdrive-service-account`, upload
   the Google service-account JSON key that has access to the DVC Drive folder.
   Never commit this key.
6. Optional, build on every push: in the job enable **GitHub hook trigger for
   GITScm polling**, and add a GitHub webhook to
   `https://<public-jenkins-host>/github-webhook/` (for a local Jenkins, expose
   it with a tunnel such as ngrok).

The first build registers the job's parameters; after that use **Build with Parameters**.

### Parameters

| Parameter | Default | Effect |
|---|---|---|
| `DATA_SOURCE` | `dvc` | `dvc`: `dvc pull` from the Drive remote using the credential above. `mounted`: use `train.csv` from the folder mounted at `/datasets` (set `DATASET_DIR=<folder>` when starting compose) |
| `RUN_LABEL` | `baseline` | candidate label: MLflow run name and `candidate` tag |
| `MAX_ROWS` | *(empty)* | override `data.max_rows` for a faster build. Keep it equal to the production model's setting, or the gate compares unlike models |
| `DEPLOY` | `false` | after promotion, build `docker/Dockerfile.api` and `kubectl set image` the `toxicity-api` deployment |
| `IMAGE` | `toxicity-api` | image name used when `DEPLOY` is on |

### Stages

| Stage | What it does | Fails the build when |
|---|---|---|
| Checkout | clone the repository | |
| Setup | create a Python 3.12 venv with `uv`, install `requirements-train.txt` | |
| Unit Tests | `pytest`, results shown in the Jenkins test report | any test fails |
| Fetch Data | `dvc pull train.csv`, or link the mounted copy | data unavailable |
| Reproduce Pipeline | `dvc repro`: prepare > train > evaluate > bias_eval | any stage fails |
| Log to MLflow | log the run, register a new version, alias `@staging` | |
| Fetch Production Baseline | download `@production`'s `bias.json` (none on the first build) | |
| Fairness Gate | compare candidate vs production; on a block, tag the version `gate=blocked` | **any gated metric regresses** |
| Promote | move `@production` to the candidate | |
| Build API Image / Deploy to Kubernetes | only when `DEPLOY` is on (needs the serving Dockerfile, the K8s deployment, and the docker/kubectl CLIs in the Jenkins image) | |

`post` always archives `metrics/*.json` (including `bias.json` and
`gate_report.json`), `dvc.lock`, `params.yaml` and the test report, so a
blocked build keeps the evidence of why it was blocked. The workspace is
cleaned after every build.

## 8. DistilBERT (GPU, Google Colab)

`src/train_distilbert.py` fine-tunes `distilbert-base-uncased` with a single
logit trained by **soft-label BCE** on the continuous `target` score (the
fraction of annotators who rated the comment toxic). It uses the same
`train.csv` / `valid.csv` / `identity_eval.parquet` as the baseline, so the
two models are compared on identical data.

| Setting | Value |
|---|---|
| Model | `distilbert-base-uncased`, 1 output logit |
| Loss | `BCEWithLogitsLoss` against `target` (soft labels) |
| Data | 160,000 train rows from `prepare.py` (same as the baseline) |
| Epochs / batch / max length | 2 / 32 / 192 tokens |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01, linear schedule with 6% warmup |
| Precision | fp16 mixed precision on GPU |
| Seed | 42 |

### Run on Colab

1. Run `dvc repro` locally so `data/processed/` exists.
2. In Google Drive create `MyDrive/toxicity-mlops/data/` and upload
   `data/processed/train.csv`, `valid.csv` and `identity_eval.parquet`.
3. Open `notebooks/train_distilbert_colab.ipynb` in Colab
   (*File > Upload notebook*), select a **T4 GPU** runtime, and run all cells.
   Outputs go to `MyDrive/toxicity-mlops/distilbert_out/`.

The script also runs on any machine with a CUDA GPU:
`pip install torch transformers==5.21.0`, then
`python src/train_distilbert.py --data-dir data/processed --out-dir distilbert_out`.

### Evaluate and log

Download `distilbert_out/` from Drive into the repository root, then score it
with the **same** evaluation code as the baseline:

```bash
python src/evaluate.py  --scores distilbert_out/valid_scores.csv         --out-dir metrics/distilbert
python src/bias_eval.py --scores distilbert_out/identity_eval_scores.csv --out-dir metrics/distilbert
python src/log_mlflow.py --run-name distilbert --metrics-dir metrics/distilbert --external-model distilbert_out
```

`--scores` takes a CSV with one `score` per row of the evaluation file, in the
same order; a file with the wrong number of rows is rejected. The MLflow run
gets tag `model_family=distilbert`, its training config as `distilbert.*`
params, and the model files under the `external_model/` artifact folder. Compare
it with the baseline by selecting both runs in the MLflow UI and clicking
**Compare**. The DistilBERT run is not registered: the registry and the
fairness gate in Jenkins manage the TF-IDF model that the pipeline retrains.

## 9. Tests

```bash
pytest -q tests
```

- `tests/test_bias_metrics.py` checks every metric against values counted by
  hand on a small golden fixture.
- `tests/test_fairness_gate.py` covers promotion, blocking on each check
  (including a more accurate model that regresses one subgroup), tolerance
  edges, a missing subgroup, no production baseline, and the CLI exit codes.
- `tests/test_mlflow_registry.py` checks the logged params/metrics and the
  fetch-baseline / promote / reject registry flow against a temporary SQLite
  store (no server needed).
