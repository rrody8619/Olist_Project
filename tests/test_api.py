from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_version" in response.json()

def test_predict_endpoint_success():
    payload = {
        "order_purchase_timestamp": "2018-05-10 10:00:00",
        "total_price": 150.0,
        "total_freight": 20.0,
        "total_items": 1,
        "total_payment": 170.0,
        "payment_installments": 2,
        "review_score": 5.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["On Time", "Late"]

def test_predict_endpoint_invalid_payload():
    payload = {
        "order_purchase_timestamp": "invalid_date",
        "total_price": -50.0  # قيمة غير معقولة للسعر
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation Error