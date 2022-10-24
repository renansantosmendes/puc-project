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