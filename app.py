"""
FastAPI inference service for Civil Comments toxicity classification.

Expected artifacts directory:
    artifacts/
        word_tfidf_vectorizer.joblib
        char_tfidf_vectorizer.joblib
        toxicity_logistic_regression.joblib
        model_metadata.json

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from scipy.sparse import hstack
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

WORD_VECTORIZER_PATH = ARTIFACTS_DIR / "word_tfidf_vectorizer.joblib"
CHAR_VECTORIZER_PATH = ARTIFACTS_DIR / "char_tfidf_vectorizer.joblib"
MODEL_PATH = ARTIFACTS_DIR / "toxicity_logistic_regression.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

word_vectorizer = None
char_vectorizer = None
model = None
metadata: dict[str, Any] = {}
prediction_threshold = 0.5
model_loaded = False

HTTP_REQUESTS_TOTAL = Counter(
    "toxicity_api_requests_total",
    "Total HTTP requests handled by the toxicity API",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "toxicity_api_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)
PREDICTIONS_TOTAL = Counter(
    "toxicity_predictions_total",
    "Total toxicity predictions by predicted class",
    ["prediction"],
)


def load_artifacts() -> None:
    """Load trained vectorizers, classifier, and optional metadata."""
    global word_vectorizer, char_vectorizer, model
    global metadata, prediction_threshold, model_loaded

    required_files = [
        WORD_VECTORIZER_PATH,
        CHAR_VECTORIZER_PATH,
        MODEL_PATH,
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        model_loaded = False
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing_files)
            + ". Put the artifacts/ directory next to app.py."
        )

    word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    if METADATA_PATH.is_file():
        with METADATA_PATH.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    else:
        metadata = {}
        logger.warning("model_metadata.json missing; defaulting threshold to 0.5.")

    # Recognize common metadata keys; adjust if your notebook uses another key.
    for key in (
        "best_threshold",
        "prediction_threshold",
        "optimal_threshold",
        "threshold",
    ):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
            prediction_threshold = float(value)
            break
    else:
        prediction_threshold = 0.5

    model_loaded = True
    logger.info("Model artifacts loaded successfully.")
    logger.info("Prediction threshold: %.4f", prediction_threshold)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts when the API starts."""
    try:
        load_artifacts()
    except Exception:
        # Keep the service alive so /health can report an unhealthy state.
        logger.exception("Could not load model artifacts.")
    yield


app = FastAPI(
    title="Civil Comments Toxicity Classification API",
    description=(
        "Toxicity prediction using word and character TF-IDF features "
        "with Logistic Regression."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    comment: str = Field(
        ...,
        min_length=1,
        max_length=20000,
        description="Comment text to classify.",
        examples=["You are a wonderful person."],
    )


class PredictionResponse(BaseModel):
    prediction: str
    is_toxic: bool
    toxicity_probability: float
    threshold: float


@app.middleware("http")
async def prometheus_http_metrics(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start_time
        # Route templates avoid high-cardinality labels from arbitrary URLs.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", request.url.path)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Civil Comments Toxicity Classification API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", tags=["Monitoring"])
def health():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded. Check the artifacts/ directory and logs.",
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": "TF-IDF + Logistic Regression",
        "threshold": prediction_threshold,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: PredictionRequest):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Ensure all model artifacts are present.",
        )

    comment = request.comment.strip()
    if not comment:
        raise HTTPException(
            status_code=422,
            detail="The comment must contain at least one non-whitespace character.",
        )

    try:
        # Preserve training order: word features first, then character features.
        word_features = word_vectorizer.transform([comment])
        char_features = char_vectorizer.transform([comment])
        combined_features = hstack(
            [word_features, char_features],
            format="csr",
        )

        # Explicitly locate class 1, which represents toxic comments.
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(f"Expected toxic class 1; model classes are {classes}")

        toxic_index = classes.index(1)
        probability = float(model.predict_proba(combined_features)[0][toxic_index])
        is_toxic = probability >= prediction_threshold
        label = "toxic" if is_toxic else "non-toxic"

        PREDICTIONS_TOTAL.labels(prediction=label).inc()

        return PredictionResponse(
            prediction=label,
            is_toxic=is_toxic,
            toxicity_probability=round(probability, 6),
            threshold=round(prediction_threshold, 6),
        )
    except Exception as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check application logs for details.",
        ) from exc


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
