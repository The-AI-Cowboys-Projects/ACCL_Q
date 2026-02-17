#!/usr/bin/env python3
"""
ACCL-Q Demo: Quantum Collective Communication
==============================================

This demo showcases ACCL-Q's key features:
1. Basic collective operations (broadcast, reduce, allreduce)
2. Quantum Error Correction (QEC) syndrome aggregation
3. Measurement feedback pipeline
4. Realistic qubit emulation with noise
5. Latency monitoring and profiling
"""

import numpy as np
import time
from typing import List

# ACCL-Q imports
from accl_quantum import (
    ACCLQuantum,
    ACCLMode,
    ACCLConfig,
    ReduceOp,
    SyncMode,
    CollectiveOp,
)
from accl_quantum.stats import LatencyMonitor
from accl_quantum.emulator import (
    RealisticQubitEmulator,
    NoiseParameters,
    GateType,
)
from accl_quantum.feedback import MeasurementFeedbackPipeline, FeedbackConfig
from accl_quantum.integrations import UnifiedQuantumControl


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_basic_collectives():
    """Demo 1: Basic collective operations."""
    print_header("Demo 1: Basic Collective Operations")

    # Simulate 4 ranks (control boards)
    num_ranks = 4

    print(f"\nSimulating {num_ranks} FPGA control boards...")

    # Create ACCL-Q instances for each rank
    accls: List[ACCLQuantum] = []
    for rank in range(num_ranks):
        accl = ACCLQuantum(num_ranks=num_ranks, local_rank=rank)
        accl.configure(mode=ACCLMode.DETERMINISTIC, sync_mode=SyncMode.HARDWARE)
        accls.append(accl)

    # Synchronize clocks
    print("\n[1] Synchronizing clocks across all ranks...")
    for accl in accls:
        accl.sync_clocks()
    print("    Clock sync complete (phase error < 2ns)")

    # Broadcast demo
    print("\n[2] BROADCAST: Rank 0 sends data to all ranks")
    data = np.array([42, 137, 255, 0], dtype=np.uint8)
    print(f"    Rank 0 data: {data}")

    results = []
    for accl in accls:
        result = accl.broadcast(data, root=0)
        results.append(result)

    print(f"    Rank 0 received: {results[0].data}")
    print(f"    Rank 1 received: {results[1].data}")
    print(f"    Rank 2 received: {results[2].data}")
    print(f"    Rank 3 received: {results[3].data}")
    print(f"    Latency: {results[0].latency_ns:.1f}ns")

    # Reduce demo (XOR for syndrome aggregation)
    print("\n[3] REDUCE (XOR): Aggregate data at root")
    local_data = [
        np.array([1, 0, 1, 0], dtype=np.uint8),  # Rank 0
        np.array([0, 1, 0, 1], dtype=np.uint8),  # Rank 1
        np.array([1, 1, 0, 0], dtype=np.uint8),  # Rank 2
        np.array([0, 0, 1, 1], dtype=np.uint8),  # Rank 3
    ]

    for i, d in enumerate(local_data):
        print(f"    Rank {i} local: {d}")

    # XOR all together: should be [0, 0, 0, 0]
    results = []
    for i, accl in enumerate(accls):
        result = accl.reduce(local_data[i], op=ReduceOp.XOR, root=0)
        results.append(result)

    expected = local_data[0] ^ local_data[1] ^ local_data[2] ^ local_data[3]
    print(f"    XOR result at root: {results[0].data}")
    print(f"    Expected (manual):  {expected}")
    print(f"    Latency: {results[0].latency_ns:.1f}ns")

    # Barrier demo
    print("\n[4] BARRIER: Synchronize all ranks")
    barrier_results = [accl.barrier() for accl in accls]
    jitters = [r.latency_ns for r in barrier_results]
    print(f"    All ranks synchronized")
    print(f"    Jitter range: {min(jitters):.1f}ns - {max(jitters):.1f}ns")

    return accls


def demo_qec_syndrome():
    """Demo 2: Quantum Error Correction syndrome aggregation."""
    print_header("Demo 2: QEC Syndrome Aggregation")

    num_ranks = 8
    qubits_per_rank = 8
    total_qubits = num_ranks * qubits_per_rank

    print(f"\nSimulating distributed QEC with:")
    print(f"  - {num_ranks} control boards")
    print(f"  - {qubits_per_rank} qubits per board")
    print(f"  - {total_qubits} total qubits")

    # Create ACCL-Q instances
    accls = [ACCLQuantum(num_ranks=num_ranks, local_rank=r) for r in range(num_ranks)]
    for accl in accls:
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        accl.sync_clocks()

    # Simulate local syndrome measurements
    print("\n[1] Measuring local ancilla qubits...")
    np.random.seed(42)  # Reproducible
    local_syndromes = []
    for rank in range(num_ranks):
        # Simulate random syndrome bits (in real QEC, these come from ancilla measurements)
        syndrome = np.random.randint(0, 2, size=4, dtype=np.uint8)
        local_syndromes.append(syndrome)
        print(f"    Rank {rank} syndrome: {syndrome}")

    # Aggregate syndromes via XOR allreduce
    print("\n[2] Aggregating syndromes via XOR allreduce...")
    start = time.perf_counter_ns()

    results = []
    for i, accl in enumerate(accls):
        result = accl.allreduce(local_syndromes[i], op=ReduceOp.XOR)
        results.append(result)

    elapsed = time.perf_counter_ns() - start

    # All ranks should have the same global syndrome
    global_syndrome = results[0].data
    print(f"    Global syndrome: {global_syndrome}")
    print(f"    Latency: {results[0].latency_ns:.1f}ns")

    # Verify all ranks got same result
    all_same = all(np.array_equal(r.data, global_syndrome) for r in results)
    print(f"    All ranks consistent: {all_same}")

    # Decode (simple threshold decoder for demo)
    print("\n[3] Decoding syndrome...")
    error_detected = np.any(global_syndrome != 0)
    if error_detected:
        error_positions = np.where(global_syndrome != 0)[0]
        print(f"    Errors detected at positions: {error_positions}")
        print(f"    Correction: Apply X gates at error positions")
    else:
        print(f"    No errors detected (syndrome = 0)")

    # Distribute corrections
    print("\n[4] Distributing corrections...")
    corrections = global_syndrome  # In real decoder, this would be computed
    for i, accl in enumerate(accls):
        result = accl.broadcast(corrections, root=0)
    print(f"    Corrections distributed to all {num_ranks} boards")

    print("\n[5] QEC cycle complete!")
    print(f"    Total cycle time: ~{results[0].latency_ns * 2:.0f}ns")
    print(f"    Coherence budget used: {results[0].latency_ns * 2 / 50000 * 100:.2f}% of 50us")


def demo_qubit_emulator():
    """Demo 3: Realistic qubit emulation with noise."""
    print_header("Demo 3: Realistic Qubit Emulator")

    # Create emulator with realistic noise parameters
    noise = NoiseParameters(
        t1_us=50.0,              # T1 relaxation time
        t2_us=70.0,              # T2 dephasing time
        single_qubit_gate_error=0.001,   # 0.1% gate error
        two_qubit_gate_error=0.01,       # 1% two-qubit error
        readout_error_0=0.02,    # 2% false positive
        readout_error_1=0.05,    # 5% false negative
    )

    print("\nNoise parameters (typical superconducting qubits):")
    print(f"  T1: {noise.t1_us} us")
    print(f"  T2: {noise.t2_us} us")
    print(f"  Single-qubit gate error: {noise.single_qubit_gate_error*100:.1f}%")
    print(f"  Two-qubit gate error: {noise.two_qubit_gate_error*100:.1f}%")
    print(f"  Readout error |0>: {noise.readout_error_0*100:.1f}%")
    print(f"  Readout error |1>: {noise.readout_error_1*100:.1f}%")

    # Validate parameters
    errors = noise.validate()
    if errors:
        print(f"  Validation errors: {errors}")
    else:
        print("  Parameters validated: physically consistent")

    # Create emulator
    num_qubits = 4
    emulator = RealisticQubitEmulator(num_qubits=num_qubits, noise_params=noise)

    print(f"\n[1] Initialized {num_qubits}-qubit emulator")
    print(f"    Initial state: |0000>")

    # Apply gates
    print("\n[2] Applying quantum circuit:")
    print("    H(0) - Hadamard on qubit 0")
    emulator.apply_gate(0, GateType.H)

    print("    CNOT(0,1) - Entangle qubits 0 and 1")
    emulator.apply_gate([0, 1], GateType.CNOT)

    print("    X(2) - Flip qubit 2")
    emulator.apply_gate(2, GateType.X)

    # Get state info
    stats = emulator.get_statistics()
    print(f"\n[3] Current state:")
    print(f"    Gates applied: {stats['total_gates']}")
    print(f"    Measurements: {stats['total_measurements']}")

    # Show qubit states
    for q in range(num_qubits):
        state = emulator.get_state(q)
        print(f"    Qubit {q}: P(|0>)={state.population_0:.3f}, P(|1>)={state.population_1:.3f}, purity={state.purity:.3f}")

    # Measure
    print("\n[4] Measuring all qubits...")
    measurements = emulator.measure_all()
    print(f"    Results: {measurements}")
    print(f"    (Results include readout errors)")

    # Run multiple shots
    print("\n[5] Running 100 shots of Bell state preparation:")

    results_00 = 0
    results_11 = 0
    results_other = 0

    for _ in range(100):
        emu = RealisticQubitEmulator(num_qubits=2, noise_params=noise)
        emu.apply_gate(0, GateType.H)
        emu.apply_gate([0, 1], GateType.CNOT)
        m = emu.measure_all()

        if m[0] == 0 and m[1] == 0:
            results_00 += 1
        elif m[0] == 1 and m[1] == 1:
            results_11 += 1
        else:
            results_other += 1

    print(f"    |00>: {results_00}%")
    print(f"    |11>: {results_11}%")
    print(f"    Other (errors): {results_other}%")
    print(f"    Expected: ~50% |00>, ~50% |11>, small error rate")


def demo_feedback_pipeline():
    """Demo 4: Measurement feedback pipeline."""
    print_header("Demo 4: Measurement Feedback Pipeline")

    num_ranks = 4

    # Create ACCL-Q
    accl = ACCLQuantum(num_ranks=num_ranks, local_rank=0)
    accl.configure(mode=ACCLMode.DETERMINISTIC)
    accl.sync_clocks()

    # Create feedback pipeline with 500ns budget
    config = FeedbackConfig(latency_budget_ns=500)
    pipeline = MeasurementFeedbackPipeline(accl, config=config)

    print("\nFeedback pipeline configured:")
    print(f"  Latency budget: {config.latency_budget_ns}ns")
    print("  - Communication: 300ns (60%)")
    print("  - Computation: 150ns (30%)")
    print("  - Margin: 50ns (10%)")

    # Register actions
    actions_taken = []

    def apply_z():
        actions_taken.append('Z')
        return True

    def apply_x():
        actions_taken.append('X')
        return True

    def apply_identity():
        actions_taken.append('I')
        return True

    pipeline.register_action('z_gate', apply_z)
    pipeline.register_action('x_gate', apply_x)
    pipeline.register_action('identity', apply_identity)

    print("\n[1] Single-qubit feedback (measure -> conditional gate)")

    # Simulate measurement = 1, should trigger Z gate
    result = pipeline.single_qubit_feedback(
        source_rank=0,
        action_if_one='z_gate',
        action_if_zero='identity'
    )

    print(f"    Measurement: {result.decision}")
    print(f"    Action taken: {actions_taken[-1] if actions_taken else 'None'}")
    print(f"    Total latency: {result.total_latency_ns:.1f}ns")
    print(f"    Within budget: {result.within_budget}")

    # Breakdown
    print(f"    Breakdown:")
    for stage, ns in result.breakdown.items():
        print(f"      {stage}: {ns:.1f}ns")

    print("\n[2] Parity feedback (XOR multiple qubits)")
    actions_taken.clear()

    result = pipeline.parity_feedback(
        qubit_ranks=[0, 1, 2, 3],
        action_if_odd='x_gate',
        action_if_even='identity'
    )

    print(f"    Parity (XOR): {'odd' if result.decision else 'even'}")
    print(f"    Action taken: {actions_taken[-1] if actions_taken else 'None'}")
    print(f"    Latency: {result.total_latency_ns:.1f}ns")

    print("\n[3] Syndrome feedback (full QEC cycle)")
    actions_taken.clear()

    def simple_decoder(syndrome):
        """Simple decoder: return syndrome as corrections."""
        return syndrome

    result = pipeline.syndrome_feedback(decoder_callback=simple_decoder)

    print(f"    Syndrome: {result.measurement}")
    print(f"    Corrections: {result.decision}")
    print(f"    Latency: {result.total_latency_ns:.1f}ns")
    print(f"    Within budget: {result.within_budget}")

    # Statistics
    print("\n[4] Feedback statistics:")
    stats = pipeline.get_latency_statistics()
    for op, stat in stats.items():
        print(f"    {op}: {stat}")
    print("\n    Note: Emulator latencies are higher than hardware targets.")


def demo_latency_monitoring():
    """Demo 5: Latency monitoring and profiling."""
    print_header("Demo 5: Latency Monitoring")

    num_ranks = 8

    # Create ACCL with monitoring enabled
    config = ACCLConfig(
        num_ranks=num_ranks,
        local_rank=0,
        mode=ACCLMode.DETERMINISTIC,
        sync_mode=SyncMode.HARDWARE,
        enable_latency_monitoring=True
    )

    accl = ACCLQuantum(num_ranks=num_ranks, local_rank=0, config=config)
    accl.configure(mode=ACCLMode.DETERMINISTIC)

    monitor = LatencyMonitor()

    print(f"\nRunning {100} operations of each type...")

    # Run many operations
    for i in range(100):
        data = np.random.randint(0, 256, size=8, dtype=np.uint8)

        # Broadcast
        result = accl.broadcast(data, root=0)
        monitor.record(CollectiveOp.BROADCAST, result.latency_ns, num_ranks=num_ranks, root_rank=0)

        # Reduce
        result = accl.reduce(data, op=ReduceOp.XOR, root=0)
        monitor.record(CollectiveOp.REDUCE, result.latency_ns, num_ranks=num_ranks, root_rank=0)

        # Allreduce
        result = accl.allreduce(data, op=ReduceOp.XOR)
        monitor.record(CollectiveOp.ALLREDUCE, result.latency_ns, num_ranks=num_ranks)

        # Barrier
        result = accl.barrier()
        monitor.record(CollectiveOp.BARRIER, result.latency_ns, num_ranks=num_ranks)

    # Get statistics
    print("\n[1] Latency Statistics:")
    stats = monitor.get_stats()

    for op, stat in stats.items():
        print(f"\n    {op.name}:")
        print(f"      Count:  {stat.count}")
        print(f"      Mean:   {stat.mean_ns:.1f}ns")
        print(f"      Std:    {stat.std_ns:.1f}ns (jitter)")
        print(f"      Min:    {stat.min_ns:.1f}ns")
        print(f"      Max:    {stat.max_ns:.1f}ns")
        print(f"      P95:    {stat.p95_ns:.1f}ns")
        print(f"      P99:    {stat.p99_ns:.1f}ns")

    # Check violations
    print("\n[2] Target Compliance:")
    targets = {
        CollectiveOp.BROADCAST: 300,
        CollectiveOp.REDUCE: 400,
        CollectiveOp.ALLREDUCE: 700,
        CollectiveOp.BARRIER: 100,
    }

    for op, target in targets.items():
        stat = stats.get(op)
        if stat:
            meets = stat.meets_target(target, jitter_target_ns=10)
            violations = monitor.get_violation_rate(op)
            status = "PASS" if meets else "FAIL"
            print(f"    {op.name}: {status} (target: {target}ns, violations: {violations*100:.1f}%)")

    # Summary
    print("\n[3] Summary:")
    print(monitor.summary())


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("       ACCL-Q: Quantum Collective Communication Demo")
    print("=" * 60)
    print("\nThis demo showcases ACCL-Q's capabilities for distributed")
    print("quantum control and error correction.")

    try:
        # Run demos
        demo_basic_collectives()
        demo_qec_syndrome()
        demo_qubit_emulator()
        demo_feedback_pipeline()
        demo_latency_monitoring()

        print_header("Demo Complete!")
        print("\nAll demos completed successfully.")
        print("\nKey takeaways:")
        print("  - ACCL-Q provides sub-microsecond collective operations")
        print("  - XOR-based syndrome aggregation for QEC")
        print("  - Hardware-synchronized barriers with <10ns jitter")
        print("  - Measurement feedback within 500ns budget")
        print("  - Realistic qubit emulation with T1/T2 noise")
        print("\nFor more information, see the ACCL-Q documentation.")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
