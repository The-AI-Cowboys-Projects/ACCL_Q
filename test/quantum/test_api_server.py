"""
ACCL-Q API Server Tests
========================

Tests for all REST API endpoints covering:
- Health/root endpoints
- Cluster management
- Collective operations (broadcast, reduce, allreduce, barrier)
- QEC syndrome demo
- Qubit emulator CRUD + gate/measure
- CORS configuration
- Input validation and error handling
"""

import pytest
from api_server import _emulators, MAX_EMULATORS


# ============================================================================
# Health & Root
# ============================================================================

class TestHealth:

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "accl-q"

    def test_root_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "ACCL-Q" in data["service"]
        assert "endpoints" in data


# ============================================================================
# Cluster Management
# ============================================================================

class TestCluster:

    def test_create_cluster(self, client):
        resp = client.post("/cluster", json={"num_ranks": 4, "mode": "deterministic"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["num_ranks"] == 4

    def test_create_cluster_invalid_mode_returns_400(self, client):
        resp = client.post("/cluster", json={"num_ranks": 4, "mode": "turbo"})
        assert resp.status_code == 400
        assert "Unknown mode" in resp.json()["detail"]

    def test_get_cluster_status_empty(self, client):
        resp = client.get("/cluster")
        assert resp.status_code == 200
        assert resp.json()["initialized"] is False

    def test_get_cluster_status_after_create(self, cluster):
        resp = cluster.get("/cluster")
        assert resp.status_code == 200
        data = resp.json()
        assert data["initialized"] is True
        assert data["num_ranks"] == 4
        assert data["ranks"] == [0, 1, 2, 3]


# ============================================================================
# Broadcast
# ============================================================================

class TestBroadcast:

    def test_broadcast_no_cluster_returns_400(self, client):
        resp = client.post("/collective/broadcast", json={"data": [1, 2, 3], "root": 0})
        assert resp.status_code == 400
        assert "not initialized" in resp.json()["detail"].lower()

    def test_broadcast_success(self, cluster):
        resp = cluster.post("/collective/broadcast", json={"data": [10, 20, 30], "root": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["data"] == [10, 20, 30]

    def test_broadcast_invalid_root_returns_400(self, cluster):
        resp = cluster.post("/collective/broadcast", json={"data": [1], "root": 99})
        assert resp.status_code == 400
        assert "exceeds cluster size" in resp.json()["detail"]

    def test_broadcast_oversized_data_returns_422(self, cluster):
        resp = cluster.post("/collective/broadcast", json={"data": list(range(5000)), "root": 0})
        assert resp.status_code == 422


# ============================================================================
# Reduce
# ============================================================================

class TestReduce:

    def test_reduce_success(self, cluster):
        resp = cluster.post("/collective/reduce", json={
            "data": [1, 2, 3], "operation": "xor", "root": 0
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

    def test_reduce_invalid_root_returns_400(self, cluster):
        resp = cluster.post("/collective/reduce", json={
            "data": [1], "operation": "xor", "root": 99
        })
        assert resp.status_code == 400
        assert "exceeds cluster size" in resp.json()["detail"]

    def test_reduce_unknown_operation_returns_400(self, cluster):
        resp = cluster.post("/collective/reduce", json={
            "data": [1, 2], "operation": "multiply", "root": 0
        })
        assert resp.status_code == 400
        assert "Unknown operation" in resp.json()["detail"]


# ============================================================================
# Allreduce
# ============================================================================

class TestAllreduce:

    def test_allreduce_success(self, cluster):
        resp = cluster.post("/collective/allreduce", json={
            "data": [5, 10, 15], "operation": "add"
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_allreduce_unknown_operation_returns_400(self, cluster):
        resp = cluster.post("/collective/allreduce", json={
            "data": [1], "operation": "divide"
        })
        assert resp.status_code == 400
        assert "Unknown operation" in resp.json()["detail"]


# ============================================================================
# Barrier
# ============================================================================

class TestBarrier:

    def test_barrier_success(self, cluster):
        resp = cluster.post("/collective/barrier")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_barrier_no_cluster_returns_400(self, client):
        resp = client.post("/collective/barrier")
        assert resp.status_code == 400


# ============================================================================
# QEC Syndrome
# ============================================================================

class TestQEC:

    def test_qec_syndrome(self, client):
        resp = client.post("/qec/syndrome", json={"num_ranks": 4, "syndrome_bits": 8})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["num_ranks"] == 4
        assert len(data["global_syndrome"]) == 8
        assert len(data["local_syndromes"]) == 4

    def test_qec_clears_previous_state(self, cluster):
        # cluster fixture creates a 4-rank cluster
        # QEC should clear it and create its own
        resp = cluster.post("/qec/syndrome", json={"num_ranks": 8, "syndrome_bits": 4})
        assert resp.status_code == 200
        assert resp.json()["num_ranks"] == 8
        # Verify cluster was replaced
        status = cluster.get("/cluster").json()
        assert status["num_ranks"] == 8


# ============================================================================
# Emulator
# ============================================================================

class TestEmulator:

    def test_create_emulator(self, client):
        resp = client.post("/emulator", json={"num_qubits": 4})
        assert resp.status_code == 200
        data = resp.json()
        assert "emulator_id" in data
        assert data["num_qubits"] == 4
        # Full UUID (36 chars with hyphens), not truncated 8-char
        assert len(data["emulator_id"]) == 36

    def test_create_emulator_at_limit_returns_429(self, client):
        # Fill up to MAX_EMULATORS
        for i in range(MAX_EMULATORS):
            resp = client.post("/emulator", json={"num_qubits": 1})
            assert resp.status_code == 200, f"Failed creating emulator #{i}"
        # Next should be rejected
        resp = client.post("/emulator", json={"num_qubits": 1})
        assert resp.status_code == 429
        assert "limit" in resp.json()["detail"].lower()

    def test_apply_gate(self, emulator):
        client, emu_id = emulator
        resp = client.post(f"/emulator/{emu_id}/gate", json={
            "emulator_id": emu_id, "gate": "H", "qubit": 0
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["gate"] == "H"

    def test_apply_unknown_gate_returns_400(self, emulator):
        client, emu_id = emulator
        resp = client.post(f"/emulator/{emu_id}/gate", json={
            "emulator_id": emu_id, "gate": "FOOBAR", "qubit": 0
        })
        assert resp.status_code == 400
        assert "Unknown gate" in resp.json()["detail"]

    def test_measure_qubits(self, emulator):
        client, emu_id = emulator
        resp = client.post(f"/emulator/{emu_id}/measure")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "measurements" in data

    def test_get_emulator_state(self, emulator):
        client, emu_id = emulator
        resp = client.get(f"/emulator/{emu_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["emulator_id"] == emu_id
        assert data["num_qubits"] == 4

    def test_delete_emulator(self, emulator):
        client, emu_id = emulator
        resp = client.delete(f"/emulator/{emu_id}")
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        # Verify gone
        resp = client.get(f"/emulator/{emu_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/emulator/does-not-exist")
        assert resp.status_code == 404


# ============================================================================
# CORS
# ============================================================================

class TestCORS:

    def test_cors_no_credentials(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Wildcard origin should be reflected
        assert resp.headers.get("access-control-allow-origin") == "*"
        # Credentials must NOT be "true" (CRITICAL-1 fix)
        assert resp.headers.get("access-control-allow-credentials", "false") != "true"
