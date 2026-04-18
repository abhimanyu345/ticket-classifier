import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Mock model state directly ─────────────────────────────────────────────────
import api.main as main_module

# Build fake model state
mock_tokenizer = MagicMock()
mock_tokenizer.return_value = {
    "input_ids": torch.tensor([[1, 2, 3]]),
    "attention_mask": torch.tensor([[1, 1, 1]])
}

mock_model = MagicMock()
mock_model.config.id2label = {
    0: "Billing inquiry",
    1: "Cancellation request",
    2: "Product inquiry",
    3: "Refund request",
    4: "Technical issue"
}
mock_output = MagicMock()
mock_output.logits = torch.tensor([[0.8, 0.05, 0.05, 0.05, 0.05]])
mock_model.return_value = mock_output

# Inject mock state before tests run
main_module.model_state["tokenizer"] = mock_tokenizer
main_module.model_state["model"]     = mock_model
main_module.model_state["device"]    = "cpu"
main_module.model_state["loaded_at"] = 0.0

from api.main import app

client = TestClient(app, raise_server_exceptions=False)

# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["device"] == "cpu"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "predict" in response.json()

def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

def test_predict_returns_expected_fields():
    response = client.post("/predict", json={"text": "I need a refund"})
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert "all_scores" in data
    assert "latency_ms" in data
    assert data["confidence"] > 0