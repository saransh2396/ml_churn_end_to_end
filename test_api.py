from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Churn prediction API is running - CI/CD test"


def test_prediction():
    customer = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 5,
        "Balance": 75000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1]


def test_invalid_input():
    customer = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Male",
        "Age": "hello",
        "Tenure": 5,
        "Balance": 75000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


