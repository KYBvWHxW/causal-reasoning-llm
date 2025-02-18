import pytest
from fastapi.testclient import TestClient
from server import app, ModelType

client = TestClient(app)

def test_get_domain_schema():
    response = client.get("/api/v1/schema/economics")
    assert response.status_code == 200
    data = response.json()
    assert "schema" in data
    assert "variables" in data
    assert "C1" in data["variables"]
    assert "C2" in data["variables"]
    assert "E" in data["variables"]

def test_evaluate_causal_reasoning():
    request_data = {
        "domain": "economics",
        "query_variable": "C2",
        "c1_state": 1,
        "e_state": 1,
        "model": "claude"
    }
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert 0 <= data["probability"] <= 1
    assert "explanation" in data

def test_invalid_domain():
    response = client.get("/api/v1/schema/invalid_domain")
    assert response.status_code == 200
    data = response.json()
    assert "error" in data

def test_invalid_query_variable():
    request_data = {
        "domain": "economics",
        "query_variable": "invalid_var",
        "c1_state": 1,
        "e_state": 1
    }
    response = client.post("/api/v1/evaluate_causal_reasoning", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "error" in data
