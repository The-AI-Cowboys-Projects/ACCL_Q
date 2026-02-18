"""
ACCL-Q REST API Server for IBM Cloud Code Engine
=================================================

Provides HTTP endpoints for quantum collective operations simulation.
"""

import asyncio
import logging
import os
import time
import uuid

import numpy as np
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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
from accl_quantum.feedback import MeasurementFeedbackPipeline, FeedbackConfig, HardwareFeedbackEngine
from accl_quantum.hardware_accel import HardwareAccelerator
from accl_quantum.constants import ULLPipelineConfig, ULL_TARGET_TOTAL_NS

# Constants
MAX_EMULATORS = int(os.environ.get("ACCLQ_MAX_EMULATORS", "50"))
EMULATOR_TTL_SECONDS = int(os.environ.get("ACCLQ_EMULATOR_TTL", "3600"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("ACCLQ_RATE_LIMIT", "300"))

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
    "ultra_low_latency": ACCLMode.ULTRA_LOW_LATENCY,
}

# Global instances
_accl_instances: Dict[int, ACCLQuantum] = {}
_emulators: Dict[str, RealisticQubitEmulator] = {}
_emulator_timestamps: Dict[str, float] = {}  # emulator_id -> creation time
_rate_limit_counts: Dict[str, List[float]] = {}  # ip -> [timestamps]
_state_lock = asyncio.Lock()


def _cleanup_stale_emulators() -> int:
    """Remove emulators that have exceeded TTL. Returns count removed."""
    now = time.time()
    stale = [
        eid for eid, ts in _emulator_timestamps.items()
        if now - ts > EMULATOR_TTL_SECONDS
    ]
    for eid in stale:
        _emulators.pop(eid, None)
        _emulator_timestamps.pop(eid, None)
    if stale:
        logger.info(f"Cleaned up {len(stale)} stale emulators")
    return len(stale)


# Request/Response models
class CreateClusterRequest(BaseModel):
    num_ranks: int = Field(default=4, ge=2, le=16, description="Number of FPGA ranks to simulate")
    mode: str = Field(default="deterministic", description="Operation mode: standard, deterministic, low_latency, ultra_low_latency")


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
    logger.info("ACCL-Q API Server starting...")
    yield
    logger.info("ACCL-Q API Server shutting down...")
    _accl_instances.clear()
    _emulators.clear()
    _emulator_timestamps.clear()
    _rate_limit_counts.clear()


app = FastAPI(
    title="ACCL-Q API",
    description="Quantum Collective Communication Emulator API with Ultra-Low-Latency support",
    version="0.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple in-memory rate limiter per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - 60

    # Get or create timestamp list for this IP
    timestamps = _rate_limit_counts.get(client_ip, [])
    # Remove timestamps outside the 1-minute window
    timestamps = [t for t in timestamps if t > window_start]

    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again later."},
        )

    timestamps.append(now)
    _rate_limit_counts[client_ip] = timestamps
    return await call_next(request)


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
        "version": "0.3.0",
        "endpoints": {
            "/health": "Health check",
            "/cluster": "Create/manage ACCL cluster",
            "/collective/broadcast": "Broadcast operation",
            "/collective/reduce": "Reduce operation",
            "/collective/allreduce": "Allreduce operation",
            "/collective/barrier": "Barrier synchronization",
            "/qec/syndrome": "QEC syndrome aggregation demo",
            "/ull/status": "ULL pipeline status",
            "/ull/feedback": "Run ULL autonomous feedback cycle",
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
    async with _state_lock:
        instances = dict(_accl_instances)

    if not instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized. POST /cluster first.")

    if request.root >= len(instances):
        raise HTTPException(status_code=400, detail=f"Root rank {request.root} exceeds cluster size")

    data = np.array(request.data, dtype=np.uint8)

    # Execute on all ranks
    results = []
    for rank, accl in instances.items():
        result = accl.broadcast(data, root=request.root)
        results.append(result)

    return OperationResult(
        success=all(r.success for r in results),
        data=results[0].data.tolist() if results[0].data is not None else None,
        latency_ns=results[0].latency_ns,
        message=f"Broadcast from rank {request.root} to {len(instances)} ranks"
    )


@app.post("/collective/reduce", response_model=OperationResult)
async def reduce(request: ReduceRequest):
    """Execute reduce operation."""
    async with _state_lock:
        instances = dict(_accl_instances)

    if not instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    if request.root >= len(instances):
        raise HTTPException(status_code=400, detail=f"Root rank {request.root} exceeds cluster size")

    op = OP_MAP.get(request.operation)
    if op is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operation: {request.operation}. Valid: {', '.join(OP_MAP)}"
        )

    data = np.array(request.data, dtype=np.uint8)

    # Execute on root rank
    accl = instances[request.root]
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
    async with _state_lock:
        instances = dict(_accl_instances)

    if not instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    op = OP_MAP.get(request.operation)
    if op is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown operation: {request.operation}. Valid: {', '.join(OP_MAP)}"
        )

    data = np.array(request.data, dtype=np.uint8)

    # Execute on rank 0
    accl = instances[0]
    result = accl.allreduce(data, op=op)

    return OperationResult(
        success=result.success,
        data=result.data.tolist() if result.data is not None else None,
        latency_ns=result.latency_ns,
        message=f"Allreduce ({request.operation}) across {len(instances)} ranks"
    )


@app.post("/collective/barrier", response_model=OperationResult)
async def barrier():
    """Execute barrier synchronization."""
    async with _state_lock:
        instances = dict(_accl_instances)

    if not instances:
        raise HTTPException(status_code=400, detail="Cluster not initialized")

    results = []
    for accl in instances.values():
        result = accl.barrier()
        results.append(result)

    latencies = [r.latency_ns for r in results]

    return OperationResult(
        success=all(r.success for r in results),
        latency_ns=sum(latencies) / len(latencies),
        message=f"Barrier synchronized {len(instances)} ranks, jitter: {max(latencies) - min(latencies):.1f}ns"
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
        # Clean up stale emulators before checking limit
        _cleanup_stale_emulators()

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
        _emulator_timestamps[emulator_id] = time.time()

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
    async with _state_lock:
        emulator = _emulators.get(emulator_id)
    if emulator is None:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

    # Validate qubit indices against emulator size
    if request.qubit >= emulator.num_qubits:
        raise HTTPException(
            status_code=400,
            detail=f"Qubit {request.qubit} out of range (emulator has {emulator.num_qubits} qubits)"
        )
    if request.target is not None and request.target >= emulator.num_qubits:
        raise HTTPException(
            status_code=400,
            detail=f"Target qubit {request.target} out of range (emulator has {emulator.num_qubits} qubits)"
        )

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
    async with _state_lock:
        emulator = _emulators.get(emulator_id)
    if emulator is None:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")

    if qubits is None:
        results = emulator.measure_all()
    else:
        for q in qubits:
            if q < 0 or q >= emulator.num_qubits:
                raise HTTPException(
                    status_code=400,
                    detail=f"Qubit index {q} out of range [0, {emulator.num_qubits})"
                )
        results = [emulator.measure(q) for q in qubits]

    return {
        "success": True,
        "measurements": results,
        "statistics": emulator.get_statistics()
    }


@app.get("/emulator/{emulator_id}")
async def get_emulator_state(emulator_id: str):
    """Get emulator state."""
    async with _state_lock:
        emulator = _emulators.get(emulator_id)
    if emulator is None:
        raise HTTPException(status_code=404, detail=f"Emulator {emulator_id} not found")
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
        _emulator_timestamps.pop(emulator_id, None)
    return {"success": True, "message": f"Emulator {emulator_id} deleted"}


# ULL Pipeline Endpoints
_ull_engine: Optional[HardwareFeedbackEngine] = None


class ULLConfigRequest(BaseModel):
    syndrome_bits: int = Field(default=16, ge=1, le=512)
    coherence_time_us: float = Field(default=50.0, gt=0)
    fiber_length_m: float = Field(default=1.0, ge=0)


@app.get("/ull/status")
async def ull_status():
    """Get ULL pipeline status."""
    if _ull_engine is None:
        return {
            "initialized": False,
            "message": "ULL engine not initialized. POST /ull/configure first."
        }

    stats = _ull_engine.get_stats()
    return {
        "initialized": True,
        "armed": _ull_engine._armed,
        "stats": stats,
        "latency_budget_ns": ULL_TARGET_TOTAL_NS,
    }


@app.post("/ull/configure")
async def ull_configure(request: ULLConfigRequest):
    """Configure and arm the ULL hardware feedback pipeline."""
    global _ull_engine

    config = ULLPipelineConfig(
        max_syndrome_bits=request.syndrome_bits,
        coherence_time_us=request.coherence_time_us,
        fiber_length_m=request.fiber_length_m,
    )

    _ull_engine = HardwareFeedbackEngine(config)

    def simple_decoder(syndrome):
        return syndrome

    entries = _ull_engine.program_pipeline(
        decoder_fn=simple_decoder,
        syndrome_bits=request.syndrome_bits,
    )

    return {
        "success": True,
        "lut_entries": entries,
        "estimated_latency_ns": _ull_engine._hw_accel.estimate_latency_ns(),
        "message": f"ULL pipeline armed with {entries} LUT entries"
    }


@app.post("/ull/feedback")
async def ull_feedback(num_cycles: int = 1):
    """Run ULL autonomous feedback cycle(s)."""
    if num_cycles < 1 or num_cycles > 10000:
        raise HTTPException(status_code=400, detail="num_cycles must be between 1 and 10000")
    if _ull_engine is None:
        raise HTTPException(
            status_code=400,
            detail="ULL engine not initialized. POST /ull/configure first."
        )

    if num_cycles == 1:
        result = _ull_engine.run_autonomous_cycle()
        return {
            "success": result.success,
            "total_latency_ns": result.total_latency_ns,
            "within_budget": result.within_budget,
            "phases": result.phases,
        }
    else:
        results = _ull_engine.run_continuous(num_cycles=min(num_cycles, 1000))
        violations = sum(1 for r in results if not r.within_budget)
        latencies = [r.total_latency_ns for r in results]
        return {
            "success": all(r.success for r in results),
            "num_cycles": len(results),
            "violations": violations,
            "mean_latency_ns": float(np.mean(latencies)),
            "max_latency_ns": float(np.max(latencies)),
            "min_latency_ns": float(np.min(latencies)),
        }


@app.post("/ull/disarm")
async def ull_disarm():
    """Disarm the ULL pipeline."""
    global _ull_engine
    if _ull_engine is None:
        raise HTTPException(status_code=400, detail="ULL engine not initialized")
    _ull_engine.disarm()
    return {"success": True, "message": "ULL pipeline disarmed"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
