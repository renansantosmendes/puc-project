import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from fastapi.testclient import TestClient
from main import app, model, scaler

client = TestClient(app)


def test_home():
    response = client.get('/')
    assert response.json() == {"Hello": "World"}


def test_model():
    assert isinstance(model, RandomForestClassifier)


def test_scaler():
    assert isinstance(scaler, StandardScaler)