import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200


def test_prediction_endpoint(client):
    response = client.post(
        "/predict",
        json={"comment": "I really like this product"}
    )

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result, dict)


def test_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
