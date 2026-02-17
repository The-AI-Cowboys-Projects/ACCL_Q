"""Shared fixtures for ACCL-Q API server tests."""

import sys
import os

import pytest

# Ensure the driver package and api_server are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient
from api_server import app, _accl_instances, _emulators


@pytest.fixture()
def client():
    """TestClient that clears global state between tests."""
    _accl_instances.clear()
    _emulators.clear()
    with TestClient(app) as c:
        yield c
    _accl_instances.clear()
    _emulators.clear()


@pytest.fixture()
def cluster(client):
    """Create a 4-rank deterministic cluster and return the client."""
    resp = client.post("/cluster", json={"num_ranks": 4, "mode": "deterministic"})
    assert resp.status_code == 200
    return client


@pytest.fixture()
def emulator(client):
    """Create an emulator instance and return (client, emulator_id)."""
    resp = client.post("/emulator", json={"num_qubits": 4})
    assert resp.status_code == 200
    emulator_id = resp.json()["emulator_id"]
    return client, emulator_id
