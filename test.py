import json
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient
from main import app, scaler, model

client = TestClient(app)


def test_home():
    response = client.get('/')
    assert response.json() == {"Hello": "World"}


def test_models():
    assert isinstance(scaler, StandardScaler)
    assert isinstance(model, RandomForestClassifier)


def test_predict():
    data = json.dumps({
        "baseline_value": 120.0,
        "accelerations": 0.0,
        "fetal_movement": 0.0,
        "uterine_contractions": 0.0,
        "light_decelerations": 0.0,
        "severe_decelerations": 0.0,
        "prolongued_decelerations": 0.0
    })

    response = client.post(
        '/api/predict',
        headers={"Content-Type": "application/json"},
        data=data,
    )
    print(response.json())
    assert isinstance(response.json()['prediction'], float)
