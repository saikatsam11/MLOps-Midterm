# Model training, MLflow and Jenkins: reproduction guide

This guide reproduces the modelling part of the project step by step:

1. train the TF-IDF + logistic regression model and measure its bias with DVC,
2. run the same pipeline in Jenkins, tracked and registered in MLflow,
3. show the **fairness gate** blocking a more accurate but less fair model,
4. (optional) the hyperparameter search and the DistilBERT model.

Commands are given for **Windows PowerShell** and **Linux / macOS**. Run every
command from the repository root unless a step says otherwise.

> Content warning: the dataset contains abusive and offensive text.

---

## Before you start

Install these first:

| Tool | Version | Check with |
|---|---|---|
| Git | any recent | `git --version` |
| Python | **3.12** | Windows: `py -3.12 --version` / Linux, macOS: `python3.12 --version` |
| Docker Desktop (or Docker Engine + Compose) | any recent | `docker compose version` |

You also need a Google account that has been given access to the project's
DVC data folder on Google Drive.

---

## Step 1. Clone the repository and install

**Windows (PowerShell)**

```powershell
git clone https://github.com/saikatsam11/MLOps-Midterm.git
cd MLOps-Midterm
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that window
and try again.

**Linux / macOS**

```bash
git clone https://github.com/saikatsam11/MLOps-Midterm.git
cd MLOps-Midterm
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
```

**Check:** run the unit tests.

```bash
pytest -q tests
```

You should see `34 passed`. Keep this terminal open with the virtual
environment active (the prompt starts with `(.venv)`) for the next steps.

---

## Step 2. Download the data

```bash
dvc pull
```

A browser window opens: sign in with the Google account that has access to the
data and allow access. When the command finishes, `train.csv` is in the
repository root.

> Do this **before** Step 4. The Google sign-in returns to `http://localhost:8080`,
> the same port Jenkins uses; if Jenkins is already running, the sign-in cannot
> finish. If that happens, stop Jenkins
> (`docker compose -f jenkins/docker-compose.yml stop`), run `dvc pull`, then
> start it again.

---

## Step 3. Run the training pipeline

```bash
dvc repro
```

This runs four stages: `prepare` > `train` > `evaluate` > `bias_eval`.
The last lines printed should be:

```
Overall AUC: 0.9047
Final bias-aware score: 0.8624
```

followed by a table with the bias metrics of each identity subgroup. The
results are written to `metrics/metrics.json` (overall metrics) and
`metrics/bias.json` (bias metrics). Both are committed to the repository, and
`git diff metrics/` should show no changes: the pipeline is deterministic.

---

## Step 4. Start Jenkins and MLflow

Jenkins and the MLflow tracking server run together in Docker.

**4a. Give Docker enough memory (at least 11 GB).** The fairness gate demo in
Step 7 trains on 400,000 rows.

- **Windows:** Docker Desktop runs inside WSL2, which only gets half of the
  computer's RAM by default. Create the file `C:\Users\<your user>\.wslconfig`
  with this content:

  ```ini
  [wsl2]
  memory=11GB
  ```

  Quit Docker Desktop, run `wsl --shutdown` in PowerShell, and start Docker
  Desktop again.
- **macOS / Linux Docker Desktop:** *Settings > Resources > Memory*, set 11 GB
  or more, *Apply & restart*.

Check: `docker info --format "{{.MemTotal}}"` should print a number of at least
`11000000000`.

**4b. Start the containers.**

```bash
docker compose -f jenkins/docker-compose.yml up -d --build
```

The first start builds two images. When it finishes:

| Open | URL |
|---|---|
| Jenkins | <http://localhost:8080> |
| MLflow | <http://localhost:5000> |

If port 8080 is already used by something else, start with another port instead:
PowerShell `$env:JENKINS_PORT=8081; docker compose -f jenkins/docker-compose.yml up -d --build`,
Linux / macOS `JENKINS_PORT=8081 docker compose -f jenkins/docker-compose.yml up -d --build`,
and open <http://localhost:8081>.

---

## Step 5. Set up Jenkins (one time)

**5a. Unlock Jenkins.** Get the initial admin password:

```bash
docker exec jenkins-toxicity cat /var/jenkins_home/secrets/initialAdminPassword
```

Open Jenkins, paste the password, choose **Install suggested plugins**, wait for
the plugins to install, and create an admin user.

**5b. Add the data credential.** Jenkins cannot open a browser to sign in to
Google, so it reuses the login that `dvc pull` saved in Step 2. Find that file:

- **Windows:** press **Win + R**, enter `%LOCALAPPDATA%\pydrive2fs\Cache` and
  press Enter. Open the folder inside it (its name ends in
  `.apps.googleusercontent.com`); the file is `default.json`.
- **Linux:** `~/.cache/pydrive2fs/<folder>/default.json`
- **macOS:** `~/Library/Caches/pydrive2fs/<folder>/default.json`

In Jenkins: **Manage Jenkins > Credentials > System > Global credentials
(unrestricted) > Add Credentials**, then:

| Field | Value |
|---|---|
| Kind | **Secret file** |
| File | the `default.json` above (on Windows, paste its full path into the file dialog's *File name* box; `AppData` is hidden) |
| ID | `gdrive-user-credentials` |

Click **Create**. The pipeline never prints this file's contents in the build
log. It gives access to that Google account's Drive, so keep it only in
Jenkins.

**5c. Create the pipeline job.** **+ New Item**, name `toxicity-cd`, select
**Pipeline**, **OK**. In the **Pipeline** section:

| Field | Value |
|---|---|
| Definition | Pipeline script from SCM |
| SCM | Git |
| Repository URL | `https://github.com/saikatsam11/MLOps-Midterm.git` |
| Branch Specifier | `*/main` |
| Script Path | `Jenkinsfile` |

Click **Save**.

---

## Step 6. Run the pipeline in Jenkins

Open the `toxicity-cd` job and click **Build Now** (the first build always uses
the default parameters; from then on the button is **Build with Parameters**).

The build runs these stages: Checkout > Setup > Unit Tests > Fetch Data >
Reproduce Pipeline > Log to MLflow > Fetch Production Baseline > Fairness Gate >
Promote. It should finish green. In the build's **Console Output** look for:

```
Overall AUC: 0.9047
Final bias-aware score: 0.8624
Registered toxicity-classifier version 1 -> alias 'staging'
No production baseline found; first model is promoted.
FAIRNESS GATE PASSED.
Promoted toxicity-classifier v1 to 'production' (was none).
```

The same numbers as Step 3, now produced in a clean Linux container.

**See it in MLflow** (<http://localhost:5000>):

- **Experiments > `toxicity-bias`**: the run with all parameters, about 50
  metrics (including `<identity>.bpsn_auc` for every subgroup), the threshold
  sweep curves `sweep_f1` / `sweep_precision` / `sweep_recall`, and the
  artifacts (metric files and the model).
- **Models > `toxicity-classifier`**: version 1 with the aliases `@production`
  and `@staging`.

**Automatic builds.** After the first build, Jenkins checks the repository every
5 minutes (`pollSCM` in the `Jenkinsfile`). When a new commit lands on the
branch the job watches, a build starts by itself with the default parameters,
and the build page shows **Started by an SCM change**. You can see the polling
log under **Git Polling Log** in the job's menu.

---

## Step 7. Fairness gate demo

Two new candidate models are trained on **twice as much data** (400,000 rows)
and checked by the fairness gate against the production model from Step 6:

| Candidate | `CANDIDATE` | What changes in training |
|---|---|---|
| A | `imbalanced` | half of the **non-toxic** comments that mention Muslims are left out, so the model sees that identity mostly in toxic comments |
| B | `reweighted` | **non-toxic** comments that mention any identity count twice as much in training |

Run **Build with Parameters** twice, waiting for the first build to finish
before starting the second:

| Build | `CANDIDATE` | `MAX_ROWS` | Expected result |
|---|---|---|---|
| Candidate A | `imbalanced` | `400000` | **red**, stopped at *Fairness Gate* |
| Candidate B | `reweighted` | `400000` | **green**, promoted to `@production` |

Leave the other parameters at their defaults.

**What to look for:**

- **Candidate A's Console Output** shows the gate's check table. Overall AUC and
  the final score went **up**, but one subgroup went down:

  ```
  muslim.bpsn_auc            0.7244   0.7751  -0.0507  -0.0100  FAIL
  FAIRNESS GATE BLOCKED: 2 check(s) regressed: muslim.bpsn_auc, transgender.subgroup_auc
  ```

  In MLflow, **Models > `toxicity-classifier`**: this version is tagged
  `gate=blocked` with the failed checks in `gate_failed`. The build page's
  **Build Artifacts** keep `metrics/gate_report.json` and `metrics/bias.json`.
- **Candidate B's Console Output** ends with `FAIRNESS GATE PASSED.` and
  `Promoted toxicity-classifier v3 to 'production' (was v1).`

The numbers are in [Results](#results).

**To run the demo again**, first move `@production` back to version 1 (a
registry rollback):

```bash
python -c "from mlflow import MlflowClient; MlflowClient('http://127.0.0.1:5000').set_registered_model_alias('toxicity-classifier', 'production', '1')"
```

---

## Step 8 (optional). Hyperparameter search in MLflow

With the containers from Step 4 running and the virtual environment active:

```bash
python src/tune.py
```

It trains 6 logistic regression variants (`C` = 0.5, 1, 4 x `class_weight` =
balanced, none) on a 40,000-row subsample and prints a results table.

In MLflow open **Experiments > `toxicity-tuning`**, expand the parent run
`tfidf-grid-search`, select the 6 child runs, click **Compare**, and choose
**Parallel Coordinates** with `C`, `class_weight`, `valid_roc_auc` and
`min_bpsn_auc`. The variant with the best validation AUC is not the one with the
best worst-case BPSN: accuracy and fairness pull in different directions.

---

## Step 9 (optional). DistilBERT

DistilBERT needs a GPU, so it is trained on Google Colab. Its evaluated results
are already committed in `metrics/distilbert/` and summarised in
[Results](#results); these steps retrain it.

1. After Step 3, upload `data/processed/train.csv`, `data/processed/valid.csv`
   and `data/processed/identity_eval.parquet` to a Google Drive folder named
   `toxicity-mlops/data` (inside *My Drive*).
2. Open <https://colab.research.google.com>, **File > Upload notebook**, choose
   `notebooks/train_distilbert_colab.ipynb`.
3. **Runtime > Change runtime type > T4 GPU** (any GPU works), then
   **Runtime > Run all**. Allow Colab to access Google Drive when asked.
4. When it finishes, download the folder `My Drive/toxicity-mlops/distilbert_out`
   into the repository root, so that `distilbert_out/valid_scores.csv` exists.
5. Score it with the same evaluation code as the baseline, and log it to MLflow
   (containers from Step 4 running):

   ```bash
   python src/evaluate.py  --scores distilbert_out/valid_scores.csv         --out-dir metrics/distilbert
   python src/bias_eval.py --scores distilbert_out/identity_eval_scores.csv --out-dir metrics/distilbert
   python src/log_mlflow.py --run-name distilbert --metrics-dir metrics/distilbert --external-model distilbert_out
   ```

   In MLflow **Experiments > `toxicity-bias`**, select the `distilbert` run and a
   baseline run and click **Compare**.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `dvc pull` sign-in never completes | Jenkins is using port 8080. Stop it (`docker compose -f jenkins/docker-compose.yml stop`), run `dvc pull`, start it again. |
| Jenkins build fails at *Fetch Data* with `Could not find credentials entry with ID 'gdrive-user-credentials'` | Step 5b was skipped or the ID is misspelled. |
| Jenkins build fails at *Fetch Data* with an `SSL` error | A dropped connection to Google. Run the build again. |
| Jenkins build fails at *Reproduce Pipeline* with `exited with 137` | Docker ran out of memory. Do Step 4a. |
| **Build with Parameters** does not show `CANDIDATE` | Jenkins reads the parameters during the first build. Run one build with the defaults first. |
| A second MLflow server will not start on port 5000 | The Jenkins stack already serves MLflow there; use <http://localhost:5000>. |
| Stop everything | `docker compose -f jenkins/docker-compose.yml down` (add `-v` to also delete all Jenkins and MLflow data). |

---

## Results

### Baseline vs DistilBERT

Same training data (160,000 rows), same validation set (40,000 rows), same
identity hold-out (100,000 rows), same evaluation code.

| Metric | TF-IDF + LR | DistilBERT |
|---|---|---|
| Validation ROC-AUC | 0.929 | **0.966** |
| Best validation F1 | 0.614 | **0.692** |
| Identity hold-out AUC | 0.905 | **0.946** |
| Final bias-aware score | 0.862 | **0.914** |
| Lowest subgroup BPSN AUC | 0.734 (white) | 0.822 (homosexual_gay_or_lesbian) |

DistilBERT is better on every metric and every subgroup, but bias is reduced,
not removed: BPSN is still its weakest metric.

### Fairness gate demo (Step 7)

All three models are scored on the same 100,000-row identity hold-out:

| Model | Overall AUC | Final score | muslim BPSN | Lowest BPSN | Gate |
|---|---|---|---|---|---|
| Production (standard, 200k rows) | 0.9047 | 0.8624 | 0.775 | 0.734 (white) | |
| A: imbalanced, 400k rows | **0.9101 (+0.0054)** | **0.8670 (+0.0046)** | **0.724 (-0.051)** | 0.724 (muslim) | **BLOCKED**: `muslim.bpsn_auc` -0.051, `transgender.subgroup_auc` -0.011 |
| B: reweighted, 400k rows | 0.9066 (+0.0019) | 0.8692 (+0.0068) | 0.812 (+0.037) | 0.776 (black) | **PASSED**: BPSN improves for all 12 subgroups |

Candidate A is better on overall AUC and on the final score, yet the gate blocks
it because one subgroup regresses. Its second failed check,
`transgender.subgroup_auc`, is on a 630-row subgroup close to the 500-row
cutoff, where AUCs are noisy.

---

## How it works (reference)

### Pipeline (`dvc.yaml`)

| Stage | Script | Output |
|---|---|---|
| `prepare` | `src/prepare.py` | `data/processed/train.csv`, `valid.csv`, `identity_eval.parquet` |
| `train` | `src/train.py` | TF-IDF vectorizers + logistic regression in `artifacts/` |
| `evaluate` | `src/evaluate.py` | `metrics/metrics.json`: overall AUC, F1, threshold sweep |
| `bias_eval` | `src/bias_eval.py` | `metrics/bias.json`: per-identity bias metrics |

All settings are in `params.yaml`. `dvc repro` reruns only the stages whose
code, data or parameters changed.

**Identity hold-out.** Identity labels exist only on a subset of comments.
`prepare.py` first samples `data.identity_eval_size` (100,000) identity-labelled
rows with a fixed seed into `identity_eval.parquet` and removes them *before*
the training sample is drawn, so bias metrics are always computed on comments
no model was trained on, and are comparable across runs and training sizes.

**Training strategies** (`candidate.strategy` in `params.yaml`): `standard`
(the production recipe), `imbalanced` and `reweighted` (Step 7). Identity
mentions are found by whole-word matching against the term lists in
`params.yaml`.

### Bias metrics

From Borkan et al., *Nuanced Metrics for Measuring Unintended Bias with Real
Data for Text Classification* (arXiv:1903.04561). A comment belongs to an
identity subgroup when its identity score is `>= 0.5`.

| Metric | Computed on | A low value means |
|---|---|---|
| Subgroup AUC | comments mentioning the identity | the model cannot rank toxicity within the subgroup |
| BPSN AUC | non-toxic subgroup + toxic background comments | non-toxic identity mentions are over-flagged (false positives) |
| BNSP AUC | toxic subgroup + non-toxic background comments | toxic identity comments are missed (false negatives) |

```
final score = 0.25 * overall_auc + 0.25 * M(subgroup_auc) + 0.25 * M(bpsn_auc) + 0.25 * M(bnsp_auc)
M(x)        = ( mean over subgroups of x_s ^ p ) ^ (1/p),   p = -5
```

The negative power makes the worst subgroup dominate. Subgroups with fewer
than 500 evaluation rows (`bias.min_subgroup_size`) are listed under
`excluded_subgroups` in `bias.json` and left out of the score and the gate.

`metrics/bias.json`:

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

### Fairness gate (`scripts/fairness_gate.py`)

Compares the candidate's `bias.json` with the production model's:

```
PROMOTE if
    overall_auc                >= production - 0.002
    final_score                >= production
    subgroup_auc, every group  >= production - 0.01
    bpsn_auc,     every group  >= production - 0.01
BLOCK otherwise
```

Tolerances are in the `gate` section of `params.yaml`. A subgroup present in
production but missing from the candidate also blocks; BNSP is reported but not
gated. Exit code `0` = promote, `1` = block, so the Jenkins stage fails on a
block. Every check is printed and saved to `metrics/gate_report.json` before
the decision. With no production model yet, the candidate is promoted.

Run it by hand:

```bash
python scripts/fairness_gate.py --candidate metrics/bias.json --production <production bias.json>
```

### MLflow

| Logged to each run | |
|---|---|
| Params | every value in `params.yaml`, flattened (e.g. `model.C`, `bias.power`) |
| Metrics | validation AUC / F1 / precision / recall, best threshold, overall AUC and final score on the identity hold-out, power means, and `<identity>.subgroup_auc` / `.bpsn_auc` / `.bnsp_auc` for every eligible subgroup |
| Curves | `sweep_precision`, `sweep_recall`, `sweep_f1` over the threshold sweep (step = threshold x 100) |
| Tags | `candidate`, `git_commit`, `data_md5`, `model_family`, `jenkins_build` |
| Artifacts | `metrics.json`, `bias.json`, `params.yaml`, `model_metadata.json`, and the model |

The model is registered as **`toxicity-classifier`**: an MLflow pyfunc that
bundles both TF-IDF vectorizers and the classifier (`src/toxicity_pyfunc.py`).
The registry uses aliases (MLflow 3 replaces stages with aliases): `@staging` is
the latest candidate, `@production` the model that passed the gate. Blocked
versions are tagged `gate=blocked` and `gate_failed`; promoted ones `gate=passed`.

Load the production model:

```python
import mlflow, pandas as pd
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.pyfunc.load_model("models:/toxicity-classifier@production")
model.predict(pd.DataFrame({"comment_text": ["I am a gay man."]}))
```

The same steps Jenkins runs can be run by hand after `dvc repro`:

```bash
python src/log_mlflow.py --run-name baseline     # log the run, register a version, alias @staging
python scripts/promote.py fetch-baseline         # download @production's bias.json to metrics/prod_bias.json
python scripts/fairness_gate.py --candidate metrics/bias.json --production metrics/prod_bias.json
python scripts/promote.py promote                # gate passed: move @production to the candidate
python scripts/promote.py reject                 # gate blocked: tag the candidate as blocked
```

### Jenkins (`Jenkinsfile`, `jenkins/`)

`jenkins/docker-compose.yml` runs Jenkins (with Python 3.12 via `uv`) and an
MLflow server (SQLite registry, artifacts on a Docker volume); Jenkins reaches
MLflow at `http://mlflow:5000`.

| Parameter | Default | Effect |
|---|---|---|
| `DATA_SOURCE` | `dvc` | `dvc`: `dvc pull` with the `gdrive-user-credentials` credential. `mounted`: use `train.csv` from a folder mounted at `/datasets` (start compose with `DATASET_DIR=<folder>`) |
| `CANDIDATE` | `standard` | training strategy: `standard`, `imbalanced` or `reweighted` |
| `RUN_LABEL` | *(empty)* | MLflow run name; empty uses the `CANDIDATE` value |
| `MAX_ROWS` | *(empty)* | override `data.max_rows`; every size is scored on the same identity hold-out |
| `DEPLOY` | `false` | after promotion, build `docker/Dockerfile.api` and `kubectl set image` the `toxicity-api` deployment |
| `IMAGE` | `toxicity-api` | image name used when `DEPLOY` is on |

| Stage | What it does | Fails the build when |
|---|---|---|
| Checkout | clone the repository | |
| Setup | Python 3.12 venv with `uv`, install `requirements-train.txt` | |
| Unit Tests | `pytest`, shown in the Jenkins test report | a test fails |
| Fetch Data | `dvc pull train.csv`, or link the mounted copy | data unavailable |
| Reproduce Pipeline | `dvc repro` | a stage fails |
| Log to MLflow | log the run, register a new version, alias `@staging` | |
| Fetch Production Baseline | download `@production`'s `bias.json` | |
| Fairness Gate | compare candidate vs production; on a block, tag the version `gate=blocked` | **a gated metric regresses** |
| Promote | move `@production` to the candidate | |
| Build API Image / Deploy to Kubernetes | only when `DEPLOY` is on | |

After every build, `metrics/*.json` (including `bias.json` and
`gate_report.json`), `dvc.lock`, `params.yaml` and the test report are archived
with the build, so a blocked build keeps the evidence of why it was blocked.

The `triggers { pollSCM('H/5 * * * *') }` block makes Jenkins poll the
repository every 5 minutes and build each new commit with the default
parameters, so every change is retrained and gated against production. Polling
works on a local Jenkins with no public URL. A Jenkins with a public URL can use
a GitHub webhook to `https://<jenkins-host>/github-webhook/` instead, which
builds immediately on push.

### DistilBERT (`src/train_distilbert.py`)

| Setting | Value |
|---|---|
| Model | `distilbert-base-uncased`, 1 output logit |
| Loss | `BCEWithLogitsLoss` against the continuous `target` score (soft labels: the fraction of annotators who rated the comment toxic) |
| Data | the same 160,000 training rows as the baseline |
| Epochs / batch / max length | 2 / 32 / 192 tokens |
| Optimizer | AdamW, lr 2e-5, weight decay 0.01, linear schedule with 6% warmup |
| Precision | fp16 mixed precision |
| Seed | 42 |

`evaluate.py` and `bias_eval.py` take `--scores`, a CSV with one `score` per row
of the evaluation file in the same order, so any model is scored with the same
code; a file with the wrong number of rows is rejected. `log_mlflow.py
--external-model` logs the run (tag `model_family=distilbert`) without
registering it: the registry and the gate manage the model that Jenkins
retrains.

### Tests (`tests/`)

| File | Covers |
|---|---|
| `test_bias_metrics.py` | every bias metric against values counted by hand on a small fixture |
| `test_fairness_gate.py` | promotion, blocking on each check (including a more accurate model that regresses one subgroup), tolerance edges, a missing subgroup, no production model, CLI exit codes |
| `test_mlflow_registry.py` | logged params/metrics and the fetch-baseline / promote / reject flow against a temporary SQLite store |
| `test_candidates.py` | identity-term matching and the three training strategies |
