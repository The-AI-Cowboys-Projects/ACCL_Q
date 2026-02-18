"""
ACCL-Q Constants and Enumerations

Defines timing parameters, operation modes, and message types for
quantum-optimized collective communication.
"""

from enum import Enum, IntEnum
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Timing Constants (all in nanoseconds unless otherwise noted)
# ============================================================================

# Clock configuration
CLOCK_PERIOD_NS = 2          # 500 MHz system clock
CLOCK_FREQ_MHZ = 500
MAX_RANKS = 16
DATA_WIDTH_BITS = 512
BYTES_PER_WORD = DATA_WIDTH_BITS // 8

# Latency targets
TARGET_P2P_LATENCY_NS = 200
TARGET_BROADCAST_LATENCY_NS = 300
TARGET_REDUCE_LATENCY_NS = 400
TARGET_ALLREDUCE_LATENCY_NS = 400
TARGET_SCATTER_LATENCY_NS = 300
TARGET_GATHER_LATENCY_NS = 300
MAX_JITTER_NS = 10
FEEDBACK_LATENCY_BUDGET_NS = 500

# Ultra-Low-Latency (ULL) timing targets
ULL_TARGET_MULTICAST_NS = 10    # Simplified Aurora for on-board/short-link
ULL_TARGET_REDUCE_NS = 4        # Combinational XOR (1-2 cycles)
ULL_TARGET_DECODE_NS = 8        # BRAM LUT decoder (4 cycles)
ULL_TARGET_TRIGGER_NS = 2       # Hardware trigger assertion (1 cycle)
ULL_TARGET_TOTAL_NS = 50        # Total feedback budget
ULL_MAX_JITTER_NS = 2
ULL_MAX_SYNDROME_BITS = 512
ULL_LUT_DECODER_DEPTH = 4096
ULL_DMA_BUFFER_ALIGNMENT = 64
ULL_DMA_BUFFER_POOL_SIZE = 16

# Component latencies
AURORA_PHY_LATENCY_NS = 40
PROTOCOL_LATENCY_NS = 80
FIBER_DELAY_NS_PER_METER = 5
DEFAULT_FIBER_LENGTH_M = 10

# Simulation model parameters
SIM_PER_HOP_LATENCY_NS = 100    # Simulated per-hop latency in tree operations
SIM_REDUCE_OVERHEAD_NS = 5      # Additional latency per reduction level
SIM_JITTER_STD_NS = 2           # Standard deviation of simulated jitter
SIM_TREE_FANOUT = 4             # Default tree fanout for latency estimation

# Clock synchronization
MAX_PHASE_ERROR_NS = 1.0
MAX_COUNTER_SYNC_ERROR_CYCLES = 2
SYNC_TIMEOUT_US = 1000
COUNTER_WIDTH_BITS = 48

# Operation timeouts
DEFAULT_OPERATION_TIMEOUT_NS = 10000
BARRIER_TIMEOUT_NS = 10000

# Quantum timing constraints
TYPICAL_T1_MIN_US = 10
TYPICAL_T1_MAX_US = 1000
TYPICAL_T2_MIN_US = 5
TYPICAL_T2_MAX_US = 500
MAX_READOUT_TIME_NS = 1000


# ============================================================================
# Enumerations
# ============================================================================

class ACCLMode(IntEnum):
    """ACCL-Q operation modes."""
    STANDARD = 0       # Standard ACCL behavior (TCP/UDP)
    DETERMINISTIC = 1  # Deterministic timing mode (Aurora-direct)
    LOW_LATENCY = 2    # Optimized for minimum latency
    ULTRA_LOW_LATENCY = 3  # Hardware-autonomous sub-50ns feedback


class ReduceOp(IntEnum):
    """Reduction operations for collective reduce."""
    XOR = 0   # Bitwise XOR - for parity/syndrome computation
    ADD = 1   # Addition - for accumulation
    MAX = 2   # Maximum - for finding max value
    MIN = 3   # Minimum - for finding min value


class SyncMode(IntEnum):
    """Synchronization modes for collective operations."""
    HARDWARE = 0   # Hardware trigger (lowest jitter, < 2ns)
    SOFTWARE = 1   # Software barrier (higher jitter, ~10-50ns)
    NONE = 2       # No synchronization (for debugging)


class QuantumMsgType(IntEnum):
    """Message types for quantum-specific operations."""
    MEASUREMENT_DATA = 0x10     # Qubit measurement results
    SYNDROME_DATA = 0x11       # QEC syndrome information
    TRIGGER_SYNC = 0x12        # Synchronized trigger request
    PHASE_CORRECTION = 0x13    # Phase correction command
    CONDITIONAL_OP = 0x14      # Conditional operation


class CollectiveOp(IntEnum):
    """Collective operation types."""
    BROADCAST = 0
    REDUCE = 1
    ALLREDUCE = 2
    SCATTER = 3
    GATHER = 4
    ALLGATHER = 5
    BARRIER = 6


class OperationStatus(IntEnum):
    """Status codes for ACCL operations."""
    SUCCESS = 0
    TIMEOUT = 1
    SYNC_ERROR = 2
    BUFFER_ERROR = 3
    RANK_ERROR = 4
    UNKNOWN_ERROR = 255


# ============================================================================
# Configuration Structures
# ============================================================================

@dataclass
class ACCLConfig:
    """Configuration for ACCL-Q initialization."""
    num_ranks: int
    local_rank: int
    mode: ACCLMode = ACCLMode.DETERMINISTIC
    sync_mode: SyncMode = SyncMode.HARDWARE
    fiber_length_m: float = DEFAULT_FIBER_LENGTH_M
    timeout_ns: int = DEFAULT_OPERATION_TIMEOUT_NS
    enable_latency_monitoring: bool = True

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if self.num_ranks < 1 or self.num_ranks > MAX_RANKS:
            raise ValueError(f"num_ranks must be 1-{MAX_RANKS}")
        if self.local_rank < 0 or self.local_rank >= self.num_ranks:
            raise ValueError(f"local_rank must be 0-{self.num_ranks-1}")
        return True


@dataclass
class ULLPipelineConfig:
    """Configuration for Ultra-Low-Latency hardware pipeline."""
    max_syndrome_bits: int = ULL_MAX_SYNDROME_BITS
    decoder_type: str = 'lut'         # 'lut' (BRAM lookup) or 'combinational'
    lut_depth: int = ULL_LUT_DECODER_DEPTH
    use_hardware_multicast: bool = True
    use_combinational_reduce: bool = True
    coherence_time_us: float = 50.0
    auto_trigger: bool = True
    bypass_monitoring: bool = True
    dma_buffer_count: int = ULL_DMA_BUFFER_POOL_SIZE
    fiber_length_m: float = 1.0       # Short links for ULL


@dataclass
class LatencyBudget:
    """Latency budget for quantum operations."""
    total_budget_ns: float
    communication_budget_ns: float
    computation_budget_ns: float
    margin_ns: float = 50.0

    @classmethod
    def for_qec_cycle(cls, coherence_time_us: float = 100.0,
                      coherence_budget_pct: float = 10.0) -> "LatencyBudget":
        """Create budget for QEC error correction cycle.

        Args:
            coherence_time_us: Qubit coherence time in microseconds
            coherence_budget_pct: Percentage of coherence time allocated
        """
        total = coherence_time_us * 1000 * (coherence_budget_pct / 100.0)
        return cls(
            total_budget_ns=total,
            communication_budget_ns=total * 0.6,
            computation_budget_ns=total * 0.3,
            margin_ns=total * 0.1
        )

    @classmethod
    def for_feedback(cls) -> "LatencyBudget":
        """Create budget for measurement feedback."""
        return cls(
            total_budget_ns=FEEDBACK_LATENCY_BUDGET_NS,
            communication_budget_ns=300,
            computation_budget_ns=150,
            margin_ns=50
        )

    @classmethod
    def for_ull_feedback(cls, coherence_time_us: float = 50.0) -> "LatencyBudget":
        """Create ultra-low-latency budget: 0.1% of coherence time.

        For 50us coherence time, budget = 50ns.
        """
        total = coherence_time_us * 1000 * 0.001  # 0.1% of coherence time
        return cls(
            total_budget_ns=total,
            communication_budget_ns=total * 0.5,
            computation_budget_ns=total * 0.4,
            margin_ns=total * 0.1
        )


# ============================================================================
# Hardware Constants
# ============================================================================

# Aurora packet header fields (matching HLS definitions)
AURORA_PKT_TYPE_DATA = 0x0
AURORA_PKT_TYPE_CONTROL = 0x1
AURORA_PKT_TYPE_SYNC = 0x2
AURORA_PKT_TYPE_ACK = 0x3
AURORA_PKT_TYPE_BARRIER = 0x4

AURORA_DEST_BROADCAST = 0xF

# Sync message markers
SYNC_MARKER = 0xAA
SYNC_MSG_COUNTER_REQ = 0x01
SYNC_MSG_COUNTER_RESP = 0x02
SYNC_MSG_PHASE_ADJ = 0x03
SYNC_MSG_COMPLETE = 0x04
