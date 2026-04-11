from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_valid():
    response = client.post("/predict", json={
        "age": 35,
        "tenure_months": 24,
        "monthly_spend": 79.99,
        "num_support_tickets": 2,
        "region": "West",
        "plan_type": "Standard",
        "payment_method": "Credit Card"
    })
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body

def test_predict_invalid():
    response = client.post("/predict", json={"age": "not-a-number"})
    assert response.status_code == 422
