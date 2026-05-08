from fastapi.testclient import TestClient

from src.service.app import app


def test_health_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_rejects_non_image() -> None:
    client = TestClient(app)
    response = client.post(
        "/predict",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_metrics_counter() -> None:
    client = TestClient(app)
    before = client.get("/metrics").json()
    client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    after = client.get("/metrics").json()
    assert after["predict_total"] == before["predict_total"] + 1
    assert after["predict_failed"] == before["predict_failed"] + 1
