"""
ACCL-Q REST API Server for IBM Cloud Code Engine
=================================================

Provides HTTP endpoints for quantum collective operations simulation.
"""

import asyncio
import os
import time
import uuid

import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ACCL-Q imports
from accl_quantum import (
    ACCLQuantum,
    ACCLMode,
    ACCLConfig,
    ReduceOp,
    SyncMode,
    CollectiveOp,
)
from accl_quantum.emulator import (
    RealisticQubitEmulator,
    NoiseParameters,
    GateType,
)
from accl_quantum.feedback import MeasurementFeedbackPipeline, FeedbackConfig

# Constants
MAX_EMULATORS = 50

OP_MAP = {
    "xor": ReduceOp.XOR,
    "add": ReduceOp.ADD,
    "max": ReduceOp.MAX,
    "min": ReduceOp.MIN,
}

MODE_MAP = {
    "standard": ACCLMode.STANDARD,
    "deterministic": ACCLMode.DETERMINISTIC,
    "low_latency": ACCLMode.LOW_LATENCY,
}

# Global instances
_accl_instances: Dict[int, ACCLQuantum] = {}
_emulators: Dict[str, RealisticQubitEmulator] = {}
_state_lock = asyncio.Lock()


# Request/Response models
class CreateClusterRequest(BaseModel):
    num_ranks: int = Field(default=4, ge=2, le=16, description="Number of FPGA ranks to simulate")
    mode: str = Field(default="deterministic", description="Operation mode: standard, deterministic, low_latency")


class BroadcastRequest(BaseModel):
    data: List[int] = Field(..., max_length=4096, description="Data to broadcast (uint8 values)")
    root: int = Field(default=0, ge=0, description="Root rank")


class ReduceRequest(BaseModel):
    data: List[int] = Field(..., max_length=4096, description="Local data (uint8 values)")
    operation: str = Field(default="xor", description="Reduce operation: xor, add, max, min")
    root: int = Field(default=0, ge=0, description="Root rank")


class AllreduceRequest(BaseModel):
    data: List[int] = Field(..., max_length=4096, description="Local data (uint8 values)")
    operation: str = Field(default="xor", description="Reduce operation: xor, add, max, min")


class QECRequest(BaseModel):
    num_ranks: int = Field(default=8, ge=2, le=16)
    syndrome_bits: int = Field(default=4, ge=1, le=64)


class EmulatorRequest(BaseModel):
    num_qubits: int = Field(default=4, ge=1, le=16)
    t1_us: float = Field(default=50.0, gt=0)
    t2_us: float = Field(default=70.0, gt=0)
    gate_error: float = Field(default=0.001, ge=0, le=1)


class GateRequest(BaseModel):
    emulator_id: str
    gate: str = Field(..., description="Gate type: H, X, Y, Z, CNOT, etc.")
    qubit: int = Field(..., ge=0)
    target: Optional[int] = Field(default=None, ge=0, description="Target qubit for 2-qubit gates")


class OperationResult(BaseModel):
    success: bool
    data: Optional[List[int]] = None
    latency_ns: float
    message: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize on startup."""
    print("ACCL-Q API Server starting...")
    yield
    print("ACCL-Q API Server shutting down...")
    _accl_instances.clear()
    _emulators.clear()


app = FastAPI(
    title="ACCL-Q API",
    description="Quantum Collective Communication Emulator API",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health & Status
@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "accl-q"}


@app.get("/")
async def root():
    """API info."""
    return {
        "service": "ACCL-Q Quantum Emulator",
        "version": "0.2.0",
        "endpoints": {
            "/health": "Health check",
            "/cluster": "Create/manage ACCL cluster",
            "/collective/broadcast": "Broadcast operation",
            "/collective/reduce": "Reduce operation",
            "/collective/allreduce": "Allreduce operation",
            "/collective/barrier": "Barrier synchronization",
            "/qec/syndrome": "QEC syndrome aggregation demo",
            "/emulator": "Create qubit emulator",
            "/emulator/{id}/gate": "Apply quantum gate",
            "/emulator/{id}/measure": "Measure qubits",
        }
    }


# Cluster Management
@app.post("/cluster")
async def create_cluster(request: CreateClusterRequest):
    """Create an ACCL cluster with specified ranks."""
    mode = MODE_MAP.get(request.mode)
    if mode is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: {request.mode}. Valid: {', '.join(MODE_MAP)}"
        )

    async with _state_lock:
        # Create instances for each rank
        _accl_instances.clear()
        for rank in range(request.num_ranks):
            accl = ACCLQuantum(num_ranks=request.num_ranks, local_rank=rank)
            accl.configure(mode=mode, sync_mode=SyncMode.HARDWARE)
            accl.sync_clocks()
            _accl_instances[rank] = accl

    return {
        "success": True,
        "num_ranks": request.num_ranks,
        "mode": request.mode,
        "message": f"Created {request.num_ranks}-rank ACCL cluster"
    }


@app.get("/cluster")
async def get_cluster_status():
    """Get cluster status."""
    if not _accl_instances:
        return {"initialized": False, "num_ranks": 0}

    return {
        "initialized": True,
        "num_ranks": len(_accl_instances),
        "ranks": list(_accl_instances.keys())
    }


# Collective Operations
@app.post("/collective/broadcast", response_model=OperationResult)
async def broadcast(request: BroadcastRequest):
    """Execute broadcast operation."""
    if not _accl_instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized. POST /cluster first.")

    if request.root >= len(_accl_instances):
        raise HTTPException(status_code=400, detail=f"Root rank {request.root} exceeds cluster size")

    data = np.array(request.data, dtype=np.uint8)

    # Execute on all ranks
    results = []
    for rank, accl in _accl_instances.items():
        result = accl.broadcast(data, root=request.root)
        results.append(result)

    return OperationResult(
        success=all(r.success for r in results),
        data=results[0].data.tolist() if results[0].data is not None else None,
        latency_ns=results[0].latency_ns,
        message=f"Broadcast from rank {request.root} to {len(_accl_instances)} ranks"
    )


@app.post("/collective/reduce", response_model=OperationResult)
async def reduce(request: ReduceRequest):
    """Execute reduce operation."""
    if not _accl_instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    if request.root >= len(_accl_instances):
        raise HTTPException(status_code=400, detail=f"Root rank {request.root} exceeds cluster size")

    op = OP_MAP.get(request.operation)
    if op is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operation: {request.operation}. Valid: {', '.join(OP_MAP)}"
        )

    data = np.array(request.data, dtype=np.uint8)

    # Execute on root rank
    accl = _accl_instances[request.root]
    result = accl.reduce(data, op=op, root=request.root)

    return OperationResult(
        success=result.success,
        data=result.data.tolist() if result.data is not None else None,
        latency_ns=result.latency_ns,
        message=f"Reduce ({request.operation}) to rank {request.root}"
    )


@app.post("/collective/allreduce", response_model=OperationResult)
async def allreduce(request: AllreduceRequest):
    """Execute allreduce operation."""
    if not _accl_instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    op = OP_MAP.get(request.operation)
    if op is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operation: {request.operation}. Valid: {', '.join(OP_MAP)}"
        )

    data = np.array(request.data, dtype=np.uint8)

    # Execute on rank 0
    accl = _accl_instances[0]
    result = accl.allreduce(data, op=op)

    return OperationResult(
        success=result.success,
        data=result.data.tolist() if result.data is not None else None,
        latency_ns=result.latency_ns,
        message=f"Allreduce ({request.operation}) across {len(_accl_instances)} ranks"
    )


@app.post("/collective/barrier", response_model=OperationResult)
async def barrier():
    """Execute barrier synchronization."""
    if not _accl_instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    results = []
    for accl in _accl_instances.values():
        result = accl.barrier()
        results.append(result)

    latencies = [r.latency_ns for r in results]

    return OperationResult(
        success=all(r.success for r in results),
        latency_ns=sum(latencies) / len(latencies),
        message=f"Barrier synchronized {len(_accl_instances)} ranks, jitter: {max(latencies) - min(latencies):.1f}ns"
    )


# QEC Demo
@app.post("/qec/syndrome")
async def qec_syndrome(request: QECRequest):
    """Run QEC syndrome aggregation demo."""
    async with _state_lock:
        # Clear previous state before creating new cluster
        _accl_instances.clear()
        for rank in range(request.num_ranks):
            accl = ACCLQuantum(num_ranks=request.num_ranks, local_rank=rank)
            accl.configure(mode=ACCLMode.DETERMINISTIC)
            accl.sync_clocks()
            _accl_instances[rank] = accl

    # Generate random syndromes with per-call RNG (no shared seed)
    rng = np.random.default_rng()
    local_syndromes = []
    for rank in range(request.num_ranks):
        syndrome = rng.integers(0, 2, size=request.syndrome_bits, dtype=np.uint8)
        local_syndromes.append(syndrome.tolist())

    # Aggregate via XOR
    start = time.perf_counter_ns()
    accl = _accl_instances[0]
    result = accl.allreduce(np.array(local_syndromes[0], dtype=np.uint8), op=ReduceOp.XOR)
    elapsed = time.perf_counter_ns() - start

    global_syndrome = result.data.tolist() if result.data is not None else []

    # Detect errors
    error_positions = [i for i, bit in enumerate(global_syndrome) if bit != 0]

    return {
        "success": True,
        "num_ranks": request.num_ranks,
        "local_syndromes": local_syndromes,
        "global_syndrome": global_syndrome,
        "errors_detected": len(error_positions) > 0,
        "error_positions": error_positions,
        "latency_ns": elapsed,
        "coherence_budget_pct": elapsed / 50000 * 100  # Assuming 50us coherence time
    }


# Qubit Emulator
@app.post("/emulator")
async def create_emulator(request: EmulatorRequest):
    """Create a qubit emulator instance."""
    async with _state_lock:
        if len(_emulators) >= MAX_EMULATORS:
            raise HTTPException(
                status_code=429,
                detail=f"Emulator limit reached ({MAX_EMULATORS}). Delete unused emulators first."
            )

        noise = NoiseParameters(
            t1_us=request.t1_us,
            t2_us=request.t2_us,
            single_qubit_gate_error=request.gate_error,
        )

        emulator_id = str(uuid.uuid4())
        emulator = RealisticQubitEmulator(num_qubits=request.num_qubits, noise_params=noise)
        _emulators[emulator_id] = emulator

    return {
        "emulator_id": emulator_id,
        "num_qubits": request.num_qubits,
        "noise_params": {
            "t1_us": request.t1_us,
            "t2_us": request.t2_us,
            "gate_error": request.gate_error,
        }
    }


@app.post("/emulator/{emulator_id}/gate")
async def apply_gate(emulator_id: str, request: GateRequest):
    """Apply a quantum gate."""
    if emulator_id not in _emulators:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

    emulator = _emulators[emulator_id]

    gate_map = {
        "H": GateType.H,
        "X": GateType.X,
        "Y": GateType.Y,
        "Z": GateType.Z,
        "S": GateType.S,
        "T": GateType.T,
        "CNOT": GateType.CNOT,
        "CZ": GateType.CZ,
    }

    gate_type = gate_map.get(request.gate.upper())
    if gate_type is None:
        raise HTTPException(status_code=400, detail=f"Unknown gate: {request.gate}")

    if gate_type in [GateType.CNOT, GateType.CZ]:
        if request.target is None:
            raise HTTPException(status_code=400, detail="Two-qubit gates require target qubit")
        emulator.apply_gate([request.qubit, request.target], gate_type)
    else:
        emulator.apply_gate(request.qubit, gate_type)

    state = emulator.get_state(request.qubit)

    return {
        "success": True,
        "gate": request.gate,
        "qubit": request.qubit,
        "state": {
            "p0": state.population_0,
            "p1": state.population_1,
            "purity": state.purity,
        }
    }


@app.post("/emulator/{emulator_id}/measure")
async def measure_qubits(emulator_id: str, qubits: Optional[List[int]] = None):
    """Measure qubits."""
    if emulator_id not in _emulators:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

    emulator = _emulators[emulator_id]

    if qubits is None:
        results = emulator.measure_all()
    else:
        results = [emulator.measure(q) for q in qubits]

    return {
        "success": True,
        "measurements": results,
        "statistics": emulator.get_statistics()
    }


@app.get("/emulator/{emulator_id}")
async def get_emulator_state(emulator_id: str):
    """Get emulator state."""
    if emulator_id not in _emulators:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

    emulator = _emulators[emulator_id]
    stats = emulator.get_statistics()

    states = {}
    for q in range(emulator.num_qubits):
        state = emulator.get_state(q)
        states[q] = {
            "p0": state.population_0,
            "p1": state.population_1,
            "purity": state.purity,
        }

    return {
        "emulator_id": emulator_id,
        "num_qubits": emulator.num_qubits,
        "qubit_states": states,
        "statistics": stats
    }


@app.delete("/emulator/{emulator_id}")
async def delete_emulator(emulator_id: str):
    """Delete emulator instance."""
    async with _state_lock:
        if emulator_id not in _emulators:
            raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

        del _emulators[emulator_id]
    return {"success": True, "message": f"Emulator {emulator_id} deleted"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
