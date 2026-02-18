"""
Tests for ULL API endpoints (/ull/*) and API server security features.
"""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi.testclient import TestClient
from api_server import (
    app, _accl_instances, _emulators, _emulator_timestamps,
    _rate_limit_counts, _ull_engine, MAX_EMULATORS,
    EMULATOR_TTL_SECONDS, RATE_LIMIT_PER_MINUTE,
)
import api_server


@pytest.fixture()
def client():
    """TestClient that clears global state between tests."""
    _accl_instances.clear()
    _emulators.clear()
    _emulator_timestamps.clear()
    _rate_limit_counts.clear()
    api_server._ull_engine = None
    with TestClient(app) as c:
        yield c
    _accl_instances.clear()
    _emulators.clear()
    _emulator_timestamps.clear()
    _rate_limit_counts.clear()
    api_server._ull_engine = None


class TestULLStatus:
    """Tests for GET /ull/status."""

    def test_status_not_initialized(self, client):
        resp = client.get("/ull/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initialized"] is False

    def test_status_after_configure(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.get("/ull/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initialized"] is True
        assert data["armed"] is True
        assert "stats" in data
        assert data["latency_budget_ns"] == 50


class TestULLConfigure:
    """Tests for POST /ull/configure."""

    def test_configure_default(self, client):
        resp = client.post("/ull/configure", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["lut_entries"] > 0
        assert "estimated_latency_ns" in data

    def test_configure_custom_syndrome_bits(self, client):
        resp = client.post("/ull/configure", json={"syndrome_bits": 32})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_configure_custom_coherence_time(self, client):
        resp = client.post("/ull/configure", json={"coherence_time_us": 100.0})
        assert resp.status_code == 200

    def test_configure_custom_fiber(self, client):
        resp = client.post("/ull/configure", json={"fiber_length_m": 2.0})
        assert resp.status_code == 200

    def test_configure_invalid_syndrome_bits(self, client):
        resp = client.post("/ull/configure", json={"syndrome_bits": 0})
        assert resp.status_code == 422  # pydantic validation

    def test_configure_syndrome_bits_too_large(self, client):
        resp = client.post("/ull/configure", json={"syndrome_bits": 600})
        assert resp.status_code == 422  # le=512 validation


class TestULLFeedback:
    """Tests for POST /ull/feedback."""

    def test_feedback_not_initialized(self, client):
        resp = client.post("/ull/feedback")
        assert resp.status_code == 400
        assert "not initialized" in resp.json()["detail"]

    def test_feedback_single_cycle(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/feedback?num_cycles=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["total_latency_ns"] <= 50
        assert data["within_budget"] is True
        assert "phases" in data

    def test_feedback_multiple_cycles(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/feedback?num_cycles=10")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["num_cycles"] == 10
        assert data["violations"] == 0
        assert data["mean_latency_ns"] <= 50

    def test_feedback_capped_at_1000(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/feedback?num_cycles=5000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_cycles"] == 1000

    def test_feedback_invalid_num_cycles(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/feedback?num_cycles=0")
        assert resp.status_code == 400

    def test_feedback_negative_num_cycles(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/feedback?num_cycles=-1")
        assert resp.status_code == 400


class TestULLDisarm:
    """Tests for POST /ull/disarm."""

    def test_disarm_not_initialized(self, client):
        resp = client.post("/ull/disarm")
        assert resp.status_code == 400

    def test_disarm_after_configure(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        resp = client.post("/ull/disarm")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_status_after_disarm(self, client):
        client.post("/ull/configure", json={"syndrome_bits": 16})
        client.post("/ull/disarm")
        resp = client.get("/ull/status")
        data = resp.json()
        assert data["initialized"] is True
        assert data["armed"] is False


class TestEmulatorTTLCleanup:
    """Tests for emulator TTL-based cleanup."""

    def test_emulator_timestamp_tracked(self, client):
        resp = client.post("/emulator", json={"num_qubits": 2})
        assert resp.status_code == 200
        eid = resp.json()["emulator_id"]
        assert eid in _emulator_timestamps
        assert _emulator_timestamps[eid] > 0

    def test_emulator_timestamp_removed_on_delete(self, client):
        resp = client.post("/emulator", json={"num_qubits": 2})
        eid = resp.json()["emulator_id"]
        client.delete(f"/emulator/{eid}")
        assert eid not in _emulator_timestamps

    def test_stale_emulators_cleaned_on_create(self, client):
        """Stale emulators are cleaned up when creating new ones."""
        # Create an emulator and mark it as stale
        resp = client.post("/emulator", json={"num_qubits": 2})
        eid = resp.json()["emulator_id"]
        _emulator_timestamps[eid] = time.time() - EMULATOR_TTL_SECONDS - 10

        # Creating another should trigger cleanup
        resp2 = client.post("/emulator", json={"num_qubits": 2})
        assert resp2.status_code == 200
        assert eid not in _emulators


class TestAPIVersion:
    """Test API version and info endpoints."""

    def test_root_version(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == "0.3.0"

    def test_root_ull_endpoints(self, client):
        resp = client.get("/")
        data = resp.json()
        assert "/ull/status" in data["endpoints"]
        assert "/ull/feedback" in data["endpoints"]

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"
