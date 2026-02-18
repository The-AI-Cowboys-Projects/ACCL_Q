"""
ACCL-Q Main Driver Class

Provides the primary interface for quantum-optimized collective
communication operations.
"""

import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass
import time
import threading

from .constants import (
    ACCLMode,
    ReduceOp,
    SyncMode,
    CollectiveOp,
    OperationStatus,
    QuantumMsgType,
    ACCLConfig,
    LatencyBudget,
    ULLPipelineConfig,
    CLOCK_PERIOD_NS,
    TARGET_BROADCAST_LATENCY_NS,
    TARGET_REDUCE_LATENCY_NS,
    MAX_JITTER_NS,
    FEEDBACK_LATENCY_BUDGET_NS,
    MAX_RANKS,
    SYNC_TIMEOUT_US,
    ULL_TARGET_MULTICAST_NS,
    ULL_TARGET_REDUCE_NS,
    ULL_TARGET_TOTAL_NS,
    ULL_MAX_SYNDROME_BITS,
    ULL_MAX_JITTER_NS,
)
from .stats import LatencyMonitor, LatencyStats, LatencyProfiler


@dataclass
class OperationResult:
    """Result of an ACCL-Q operation."""
    status: OperationStatus
    data: Optional[np.ndarray] = None
    latency_ns: float = 0.0
    timestamp_ns: int = 0

    @property
    def success(self) -> bool:
        return self.status == OperationStatus.SUCCESS


class ACCLQuantum:
    """
    ACCL-Q: Quantum-Optimized Collective Communication Driver

    This class provides the main interface for performing low-latency
    collective communication operations optimized for quantum control
    systems.

    Features:
    - Deterministic timing with hardware synchronization
    - Sub-microsecond collective operations
    - Clock synchronization across nodes
    - Latency monitoring and statistics
    - Integration with QubiC and QICK frameworks

    Example:
        accl = ACCLQuantum(num_ranks=8, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        accl.sync_clocks()

        # Broadcast measurement result
        result = accl.broadcast(measurement, root=source_rank)

        # Compute global syndrome via XOR reduction
        syndrome = accl.allreduce(local_syndrome, op=ReduceOp.XOR)
    """

    def __init__(self, num_ranks: int, local_rank: int,
                 config: Optional[ACCLConfig] = None):
        """
        Initialize ACCL-Q driver.

        Args:
            num_ranks: Total number of ranks in the system
            local_rank: This node's rank (0-indexed)
            config: Optional configuration object
        """
        if config is None:
            config = ACCLConfig(num_ranks=num_ranks, local_rank=local_rank)
        config.validate()

        self.config = config
        self.num_ranks = num_ranks
        self.local_rank = local_rank

        # State
        self._mode = ACCLMode.STANDARD
        self._sync_mode = SyncMode.HARDWARE
        self._is_initialized = False
        self._is_synchronized = False

        # Clock synchronization
        self._global_counter = 0
        self._counter_offset = 0
        self._phase_error_ns = 0.0

        # Latency monitoring
        self._monitor = LatencyMonitor() if config.enable_latency_monitoring else None
        self._latency_budget = None

        # Per-instance RNG (avoids shared global state)
        self._rng = np.random.default_rng()

        # Hardware interface (placeholder for actual FPGA interface)
        self._hw_interface = None

        # ULL hardware accelerator (lazy-initialized)
        self._hw_accel = None
        self._ull_config = None

        # Thread safety
        self._lock = threading.RLock()

    # ========================================================================
    # Configuration
    # ========================================================================

    def configure(self, mode: ACCLMode = ACCLMode.DETERMINISTIC,
                  sync_mode: SyncMode = SyncMode.HARDWARE,
                  latency_budget_ns: Optional[float] = None,
                  ull_config: Optional[ULLPipelineConfig] = None) -> None:
        """
        Configure ACCL-Q operation mode.

        Args:
            mode: Operation mode (STANDARD, DETERMINISTIC, LOW_LATENCY, ULTRA_LOW_LATENCY)
            sync_mode: Synchronization mode (HARDWARE, SOFTWARE, NONE)
            latency_budget_ns: Optional latency budget for operations
            ull_config: Configuration for ULTRA_LOW_LATENCY mode
        """
        with self._lock:
            self._mode = mode
            self._sync_mode = sync_mode

            if latency_budget_ns is not None:
                self._latency_budget = LatencyBudget(
                    total_budget_ns=latency_budget_ns,
                    communication_budget_ns=latency_budget_ns * 0.7,
                    computation_budget_ns=latency_budget_ns * 0.2,
                    margin_ns=latency_budget_ns * 0.1
                )

            if mode == ACCLMode.ULTRA_LOW_LATENCY:
                self._ull_config = ull_config or ULLPipelineConfig()
                from .hardware_accel import HardwareAccelerator
                self._hw_accel = HardwareAccelerator(self._ull_config)
                self._latency_budget = LatencyBudget.for_ull_feedback(
                    self._ull_config.coherence_time_us
                )

            self._is_initialized = True

    def set_timeout(self, timeout_ns: int) -> None:
        """Set operation timeout in nanoseconds."""
        self.config.timeout_ns = timeout_ns

    # ========================================================================
    # Clock Synchronization
    # ========================================================================

    def sync_clocks(self, timeout_us: int = SYNC_TIMEOUT_US) -> bool:
        """
        Synchronize clocks across all ranks.

        Uses NTP-like protocol to align counters with sub-nanosecond
        phase error.

        Args:
            timeout_us: Timeout for synchronization in microseconds

        Returns:
            True if synchronization successful
        """
        with self._lock:
            # In hardware implementation, this would:
            # 1. Send sync request to master
            # 2. Receive response with master's counter value
            # 3. Calculate RTT and offset
            # 4. Apply correction to local counter

            # Simulation: assume successful sync with small error
            self._counter_offset = int(self._rng.integers(-2, 3))  # +/- 2 cycles
            self._phase_error_ns = float(self._rng.uniform(-1.0, 1.0))  # +/- 1ns
            self._is_synchronized = True

            return True

    def get_global_counter(self) -> int:
        """Get current synchronized global counter value."""
        # In hardware: read from synchronized counter register
        local_counter = time.perf_counter_ns() // CLOCK_PERIOD_NS
        return local_counter + self._counter_offset

    def get_sync_status(self) -> dict:
        """Get clock synchronization status."""
        return {
            'synchronized': self._is_synchronized,
            'counter_offset_cycles': self._counter_offset,
            'phase_error_ns': self._phase_error_ns,
            'global_counter': self.get_global_counter()
        }

    # ========================================================================
    # Collective Operations
    # ========================================================================

    def broadcast(self, data: np.ndarray, root: int,
                  sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Broadcast data from root to all ranks.

        Args:
            data: Data array to broadcast (at root) or receive buffer (others)
            root: Rank that sends the data
            sync: Synchronization mode override

        Returns:
            OperationResult with received data
        """
        if self._mode == ACCLMode.ULTRA_LOW_LATENCY:
            return self._broadcast_ull(data, root)

        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            # Simulate broadcast latency
            tree_depth = int(np.ceil(np.log2(max(self.num_ranks, 2)) / np.log2(4)))
            latency = tree_depth * 100 + self._rng.normal(0, 2)  # ~100ns per hop

            # In hardware: data flows through tree
            result_data = data.copy()

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        # Record latency
        if self._monitor:
            self._monitor.record(
                CollectiveOp.BROADCAST, actual_latency,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def reduce(self, data: np.ndarray, op: ReduceOp, root: int,
               sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Reduce data to root using specified operation.

        Args:
            data: Local data to contribute
            op: Reduction operation (XOR, ADD, MAX, MIN)
            root: Rank to receive result
            sync: Synchronization mode override

        Returns:
            OperationResult with reduced data (at root)
        """
        if self._mode == ACCLMode.ULTRA_LOW_LATENCY:
            return self._reduce_ull(data, op, root)

        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            # Simulate reduction
            # In real implementation, would receive from children and combine
            result_data = data.copy()

            # Simulate tree reduce latency
            tree_depth = int(np.ceil(np.log2(max(self.num_ranks, 2)) / np.log2(4)))
            latency = tree_depth * 100 + 5  # Reduction adds ~5ns per level

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.REDUCE, actual_latency,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data if self.local_rank == root else None,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def allreduce(self, data: np.ndarray, op: ReduceOp,
                  sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Reduce and distribute result to all ranks.

        Args:
            data: Local data to contribute
            op: Reduction operation
            sync: Synchronization mode override

        Returns:
            OperationResult with reduced data (at all ranks)
        """
        if self._mode == ACCLMode.ULTRA_LOW_LATENCY:
            return self._allreduce_ull(data, op)

        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            # Allreduce = reduce + broadcast
            # In hardware: optimized implementation
            result_data = data.copy()

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.ALLREDUCE, actual_latency,
                self.num_ranks
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def scatter(self, data: Union[np.ndarray, List[np.ndarray]], root: int,
                sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Scatter different data to each rank from root.

        Args:
            data: Array of arrays (at root) - one per rank
            root: Rank that sends the data
            sync: Synchronization mode override

        Returns:
            OperationResult with this rank's portion
        """
        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            if self.local_rank == root:
                result_data = data[self.local_rank] if isinstance(data, list) else data
            else:
                # Would receive from root
                result_data = np.zeros_like(data[0] if isinstance(data, list) else data)

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.SCATTER, actual_latency,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def gather(self, data: np.ndarray, root: int,
               sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Gather data from all ranks to root.

        Args:
            data: Local data to send
            root: Rank to receive all data
            sync: Synchronization mode override

        Returns:
            OperationResult with gathered data (at root)
        """
        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            if self.local_rank == root:
                # Would receive from all ranks
                result_data = np.stack([data] * self.num_ranks)
            else:
                result_data = None

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.GATHER, actual_latency,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def allgather(self, data: np.ndarray,
                  sync: Optional[SyncMode] = None) -> OperationResult:
        """
        Gather data from all ranks to all ranks.

        Args:
            data: Local data to contribute
            sync: Synchronization mode override

        Returns:
            OperationResult with all gathered data
        """
        sync = sync if sync is not None else self._sync_mode
        start_ns = time.perf_counter_ns()

        with self._lock:
            # Would receive from all ranks
            result_data = np.stack([data] * self.num_ranks)

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.ALLGATHER, actual_latency,
                self.num_ranks
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    def barrier(self, timeout_ns: Optional[int] = None) -> OperationResult:
        """
        Synchronize all ranks with guaranteed timing.

        Uses hardware-synchronized global counter for sub-nanosecond
        release alignment.

        Args:
            timeout_ns: Operation timeout

        Returns:
            OperationResult indicating success/failure
        """
        timeout_ns = timeout_ns or self.config.timeout_ns
        start_ns = time.perf_counter_ns()

        with self._lock:
            # In hardware: wait for global counter to reach release time
            pass

        end_ns = time.perf_counter_ns()
        actual_latency = end_ns - start_ns

        if self._monitor:
            self._monitor.record(
                CollectiveOp.BARRIER, actual_latency,
                self.num_ranks
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            latency_ns=actual_latency,
            timestamp_ns=end_ns
        )

    # ========================================================================
    # Quantum-Specific Operations
    # ========================================================================

    def distribute_measurement(self, measurement: np.ndarray,
                               source_rank: int) -> OperationResult:
        """
        Distribute measurement result to all control boards.

        Optimized for measurement-based feedback where one qubit's
        measurement determines operations on other qubits.

        Args:
            measurement: Measurement outcomes array
            source_rank: Rank that performed the measurement

        Returns:
            OperationResult with measurement data
        """
        return self.broadcast(measurement, root=source_rank)

    def aggregate_syndrome(self, local_syndrome: np.ndarray) -> OperationResult:
        """
        Aggregate QEC syndrome data via XOR reduction.

        Computes global syndrome for quantum error correction
        by XORing local syndromes from all ranks.

        Args:
            local_syndrome: Local syndrome bits

        Returns:
            OperationResult with global syndrome (at all ranks)
        """
        return self.allreduce(local_syndrome, op=ReduceOp.XOR)

    def distribute_correction(self, corrections: List[np.ndarray],
                              decoder_rank: int) -> OperationResult:
        """
        Distribute decoder corrections to individual control boards.

        Args:
            corrections: Correction data for each rank
            decoder_rank: Rank running the decoder

        Returns:
            OperationResult with this rank's correction
        """
        return self.scatter(corrections, root=decoder_rank)

    def synchronized_trigger(self, trigger_time: int) -> bool:
        """
        Schedule synchronized trigger at specified global counter value.

        All ranks will trigger within < 2ns of each other.

        Args:
            trigger_time: Global counter value for trigger

        Returns:
            True if trigger scheduled successfully
        """
        current = self.get_global_counter()
        if trigger_time <= current:
            return False

        # In hardware: write trigger_time to trigger register
        # Hardware will assert trigger when counter reaches value
        return True

    # ========================================================================
    # Ultra-Low-Latency Private Methods
    # ========================================================================

    def _broadcast_ull(self, data: np.ndarray, root: int) -> OperationResult:
        """ULL broadcast: zero-copy, hardware multicast, simulated latency."""
        # Zero-copy: return data directly (no data.copy())
        # In hardware: single-cycle multicast fan-out
        result_data = data  # zero-copy identity

        # Skip monitoring when bypass_monitoring is set
        if self._ull_config and not self._ull_config.bypass_monitoring and self._monitor:
            self._monitor.record(
                CollectiveOp.BROADCAST, ULL_TARGET_MULTICAST_NS,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=ULL_TARGET_MULTICAST_NS,
            timestamp_ns=time.perf_counter_ns()
        )

    def _reduce_ull(self, data: np.ndarray, op: ReduceOp, root: int) -> OperationResult:
        """ULL reduce: validates syndrome size, zero-copy, combinational XOR."""
        data_bits = data.nbytes * 8
        if data_bits > ULL_MAX_SYNDROME_BITS:
            return OperationResult(
                status=OperationStatus.BUFFER_ERROR,
                data=None,
                latency_ns=0,
                timestamp_ns=time.perf_counter_ns()
            )

        # Zero-copy: return data directly
        result_data = data  # zero-copy identity

        if self._ull_config and not self._ull_config.bypass_monitoring and self._monitor:
            self._monitor.record(
                CollectiveOp.REDUCE, ULL_TARGET_REDUCE_NS,
                self.num_ranks, root
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=ULL_TARGET_REDUCE_NS,
            timestamp_ns=time.perf_counter_ns()
        )

    def _allreduce_ull(self, data: np.ndarray, op: ReduceOp) -> OperationResult:
        """ULL allreduce: multicast + reduce combined, zero-copy."""
        data_bits = data.nbytes * 8
        if data_bits > ULL_MAX_SYNDROME_BITS:
            return OperationResult(
                status=OperationStatus.BUFFER_ERROR,
                data=None,
                latency_ns=0,
                timestamp_ns=time.perf_counter_ns()
            )

        result_data = data  # zero-copy identity
        combined_latency = ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS

        if self._ull_config and not self._ull_config.bypass_monitoring and self._monitor:
            self._monitor.record(
                CollectiveOp.ALLREDUCE, combined_latency,
                self.num_ranks
            )

        return OperationResult(
            status=OperationStatus.SUCCESS,
            data=result_data,
            latency_ns=combined_latency,
            timestamp_ns=time.perf_counter_ns()
        )

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_latency_stats(self, operation: Optional[CollectiveOp] = None) -> dict:
        """
        Get latency statistics for operations.

        Args:
            operation: Specific operation or None for all

        Returns:
            Dictionary of operation -> LatencyStats
        """
        if self._monitor is None:
            return {}
        return {
            op.name: stats
            for op, stats in self._monitor.get_stats(operation).items()
        }

    def get_monitor(self) -> Optional[LatencyMonitor]:
        """Get the latency monitor instance."""
        return self._monitor

    def validate_timing(self) -> dict:
        """
        Validate that operations meet timing requirements.

        Uses tighter ULL targets when in ULTRA_LOW_LATENCY mode.

        Returns:
            Dictionary with validation results per operation
        """
        results = {}
        if self._monitor is None:
            return results

        if self._mode == ACCLMode.ULTRA_LOW_LATENCY:
            targets = {
                CollectiveOp.BROADCAST: ULL_TARGET_MULTICAST_NS,
                CollectiveOp.REDUCE: ULL_TARGET_REDUCE_NS,
                CollectiveOp.ALLREDUCE: ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS,
            }
            jitter_target = ULL_MAX_JITTER_NS
        else:
            targets = {
                CollectiveOp.BROADCAST: TARGET_BROADCAST_LATENCY_NS,
                CollectiveOp.REDUCE: TARGET_REDUCE_LATENCY_NS,
                CollectiveOp.ALLREDUCE: TARGET_REDUCE_LATENCY_NS,
            }
            jitter_target = MAX_JITTER_NS

        stats = self._monitor.get_stats()
        for op, target in targets.items():
            if op in stats:
                s = stats[op]
                results[op.name] = {
                    'target_ns': target,
                    'mean_ns': s.mean_ns,
                    'max_ns': s.max_ns,
                    'jitter_ns': s.std_ns,
                    'passes_latency': s.mean_ns <= target,
                    'passes_jitter': s.std_ns <= jitter_target,
                    'overall_pass': s.meets_target(target, jitter_target)
                }

        return results

    # ========================================================================
    # Context Manager Support
    # ========================================================================

    def __enter__(self):
        if not self._is_initialized:
            self.configure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        return False

    def __repr__(self):
        return (
            f"ACCLQuantum(ranks={self.num_ranks}, local_rank={self.local_rank}, "
            f"mode={self._mode.name}, sync={'yes' if self._is_synchronized else 'no'})"
        )
