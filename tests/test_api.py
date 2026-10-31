
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction_endpoint():
    response = client.post(
        "/predict",
        json={"text": "I really like this product"}
    )

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result, dict)


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
