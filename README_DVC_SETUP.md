# DVC pipeline setup (Civil Comments)

This pipeline reproduces the existing notebook workflow:
- label: `target >= 0.5`
- stratified sample up to 200,000 rows (seed 42)
- stratified 80:20 train/validation split (seed 42)
- word + character TF-IDF
- Logistic Regression
- threshold 0.5 metrics and best-F1 threshold search

## Before running

Run these commands from the project root (`Mid Term`) after copying `src/`,
`params.yaml`, and `dvc.yaml` into the repository.

Install missing dependencies in the active environment:

```bash
python -m pip install pandas numpy scipy scikit-learn joblib pyyaml
```

Important: `artifacts/` is currently tracked by Git in the original repository.
Before using it as a DVC pipeline output, untrack the existing artifact files
from Git while keeping the local copies:

```bash
git rm --cached artifacts/word_tfidf_vectorizer.joblib \
  artifacts/char_tfidf_vectorizer.joblib \
  artifacts/toxicity_logistic_regression.joblib \
  artifacts/model_metadata.json
```

Add `artifacts/` to `.gitignore` if it is not already ignored. DVC will manage
these pipeline outputs. Do not delete the local artifact files.

Then stage the pipeline definitions and commit the migration:

```bash
git add .gitignore params.yaml dvc.yaml src/
git commit -m "Add reproducible DVC training pipeline"
```

Run the pipeline:

```bash
dvc repro
dvc dag
dvc status
```

`dvc repro` retrains the model and may take several minutes. It needs enough RAM
for sparse TF-IDF matrices. The prepared CSV files and model artifacts remain
local until a DVC remote is configured and `dvc push` is run.

The evaluation writes `metrics/metrics.json`. MLflow logging can be added by
the MLflow teammate to `src/train.py` and/or `src/evaluate.py`, keeping the
DVC stage commands unchanged.
