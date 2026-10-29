# DVC Setup Guide --- Toxicity Classification MLOps

This guide explains how to configure and reproduce the updated DVC
pipeline for the toxicity-classification project.

The pipeline contains four stages:

1.  `prepare` --- prepares training, validation, and identity-evaluation
    datasets.
2.  `train` --- trains the TF-IDF + Logistic Regression model.
3.  `evaluate` --- calculates validation metrics and selects the best F1
    threshold.
4.  `bias_eval` --- evaluates overall AUC and identity-subgroup bias
    metrics.

> **Content warning:** The dataset contains abusive and offensive text.

## 1. Prerequisites

-   Git
-   Python 3.12
-   Access to the project's Google Drive DVC remote
-   The project repository:
    `https://github.com/saikatsam11/MLOps-Midterm`

Use a terminal from the repository root for the commands below.

## 2. Clone the repository and create an environment

``` bash
git clone https://github.com/saikatsam11/MLOps-Midterm.git
cd MLOps-Midterm

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pip install "dvc[gdrive]" pyarrow
```

If you already have the project's virtual environment, activate it
instead of creating a new one.

`pyarrow` is needed because the preparation stage writes
`identity_eval.parquet`. If `requirements-train.txt` already includes
these dependencies, the extra install command is harmless.

## 3. Configure Google Drive access

The DVC remote is configured in `.dvc/config` and uses the shared Google
Drive folder:

``` text
gdrive://1rfKmu9KVUZWIvy6o8j35S6OAxiFjeB7k
```

You must use a Google account that has access to that folder.

Run:

``` bash
dvc pull
```

Follow the browser sign-in and permission prompts if they appear. When
successful, DVC downloads the tracked dataset and any cached outputs
that are available remotely.

### If Google OAuth credentials are required

If DVC asks for custom Google OAuth client credentials, use the
project's approved OAuth client and configure the credentials locally
rather than committing secrets:

``` bash
dvc remote modify --local gdrive gdrive_client_id "YOUR_CLIENT_ID"
dvc remote modify --local gdrive gdrive_client_secret "YOUR_CLIENT_SECRET"
```

Replace the placeholders with your own authorized credentials. Do
**not** put client secrets, token files, or personal OAuth credentials
in Git or this README.

The shared Drive folder must also be shared with the Google account used
for authorization. If authentication fails, check the account,
Drive-folder permissions, and the local DVC configuration.

## 4. Check the dataset

The source dataset should be available at the repository root as
`train.csv`.

``` bash
ls -lh train.csv
```

If it is missing, complete the Google Drive setup and run `dvc pull`
again. Do not manually edit `train.csv.dvc`.

## 5. Reproduce the updated pipeline

Run:

``` bash
dvc repro
```

The pipeline is defined in `dvc.yaml`. DVC runs only stages whose
dependencies, parameters, or outputs require updating.

### Stage details

  ----------------------------------------------------------------------------------------
  Stage                   Script                  Main outputs
  ----------------------- ----------------------- ----------------------------------------
  `prepare`               `src/prepare.py`        `data/processed/train.csv`,
                                                  `data/processed/valid.csv`,
                                                  `data/processed/identity_eval.parquet`

  `train`                 `src/train.py`          Word and character TF-IDF vectorizers
                                                  and Logistic Regression model in
                                                  `artifacts/`

  `evaluate`              `src/evaluate.py`       `metrics/metrics.json`,
                                                  `artifacts/model_metadata.json`

  `bias_eval`             `src/bias_eval.py`      `metrics/bias.json`
  ----------------------------------------------------------------------------------------

Current default preparation settings use up to 200,000 rows, split into
160,000 training rows and 40,000 validation rows. The identity hold-out
is sampled separately using the configuration in `params.yaml`.

The current standard model uses word- and character-level TF-IDF
features with Logistic Regression. Training strategies and evaluation
settings are controlled by `params.yaml`.

## 6. Review the results

After a successful run, inspect:

``` bash
cat metrics/metrics.json
cat metrics/bias.json
```

The validation metrics include precision, recall, F1, ROC-AUC, average
precision, and the threshold selected for best F1.

The bias report includes overall AUC, a final bias-aware score, and
subgroup metrics such as subgroup AUC, BPSN AUC, and BNSP AUC. Subgroups
below the configured minimum sample size are excluded from the aggregate
score and listed separately.

Results depend on the exact code, data, and parameters. Do not assume
fixed numbers unless they are reproduced from the same project revision
and data.

## 7. Push generated DVC outputs to the remote

After `dvc repro` completes successfully, upload the generated cache
objects:

``` bash
dvc push
```

Then verify synchronization:

``` bash
dvc status -c
```

Expected successful status:

``` text
Cache and remote 'gdrive' are in sync.
```

If DVC reports missing cache objects, verify that the pipeline completed
successfully and that the correct remote is configured. Do not assume
that committing a metrics file to Git also uploads the corresponding DVC
cache objects.

## 8. Commit the DVC lock-file update

A successful reproduction may update `dvc.lock`. Review the changes:

``` bash
git status
git diff -- dvc.lock
```

If `dvc.lock` or tracked metrics changed and the changes are expected,
commit and push them:

``` bash
git add dvc.lock metrics/metrics.json metrics/bias.json
git commit -m "Update DVC pipeline and bias evaluation"
git push origin main
```

If Git reports that there is nothing to commit, do not create an empty
commit. Coordinate with teammates before committing shared changes.

**Recommended order:** run `dvc repro`, then `dvc push`, then commit and
push the updated lock file and relevant tracked metrics to Git.

## 9. Verify from a clean/synchronized workspace

To update an existing checkout and retrieve the latest DVC-tracked data
and outputs:

``` bash
git pull origin main
dvc pull
dvc status
dvc status -c
```

Expected checks:

-   `git status` shows a clean working tree when all intended Git
    changes are committed.
-   `dvc status` reports that data and pipelines are up to date.
-   `dvc status -c` reports that the local cache and remote are in sync.

A teammate needs access to the Google Drive folder and valid local
Google OAuth configuration to pull from the remote. Never share OAuth
secrets through GitHub.

## 10. Troubleshooting

### `ImportError: ... pyarrow ... required for parquet support`

Install the missing dependency in the active project environment:

``` bash
python -m pip install pyarrow
```

Then retry:

``` bash
dvc repro
```

### `dvc pull` reports missing cache files

Check the configured remote and fetch status:

``` bash
dvc remote list
dvc status -c
dvc fetch -v
```

If the latest pipeline outputs have not been pushed by anyone, reproduce
the pipeline successfully and then run `dvc push`.

### Google Drive authorization fails

-   Confirm that the Google account has access to the shared remote
    folder.
-   Complete the browser authorization flow.
-   If a custom OAuth client is used, confirm that the account is
    allowed by the OAuth application's test-user settings where
    applicable.
-   Keep client secrets and token files out of Git.

### `dvc repro` fails

Read the first traceback or error message. Fix the reported dependency
or stage failure before running `dvc push`; only push after the pipeline
has completed successfully.

------------------------------------------------------------------------

## Quick command checklist

``` bash
# From the repository root, with the project environment active
dvc pull
dvc repro
dvc push
dvc status
dvc status -c

git status
git diff -- dvc.lock
```

The DVC pipeline is ready for use when all four stages reproduce
successfully, `dvc status` reports no pipeline changes, and
`dvc status -c` confirms that the cache and remote are synchronized.