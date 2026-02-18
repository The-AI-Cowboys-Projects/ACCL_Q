"""
Tests for ACCL-Q Ultra-Low-Latency (ULL) Optimization

Validates the complete ULL pipeline: constants, hardware acceleration,
driver zero-copy paths, feedback engine, and end-to-end QEC loops.

Target: all ULL operations within 50ns (0.1% of 50us coherence time).
"""

import sys
import os
import numpy as np
import pytest

# Ensure the driver package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))

from accl_quantum.constants import (
    ACCLMode,
    ReduceOp,
    LatencyBudget,
    ULLPipelineConfig,
    ULL_TARGET_MULTICAST_NS,
    ULL_TARGET_REDUCE_NS,
    ULL_TARGET_DECODE_NS,
    ULL_TARGET_TRIGGER_NS,
    ULL_TARGET_TOTAL_NS,
    ULL_MAX_SYNDROME_BITS,
    ULL_LUT_DECODER_DEPTH,
    ULL_DMA_BUFFER_POOL_SIZE,
    ULL_MAX_JITTER_NS,
    FEEDBACK_LATENCY_BUDGET_NS,
)
from accl_quantum.hardware_accel import (
    DMABufferPool,
    LUTDecoder,
    FPGARegisterInterface,
    ULLRegister,
    HardwareAccelerator,
)
from accl_quantum.driver import ACCLQuantum, OperationResult, OperationStatus
from accl_quantum.feedback import HardwareFeedbackEngine, ULLFeedbackResult
from accl_quantum.profiler import CriticalPathProfiler
from accl_quantum.integrations import QubiCIntegration, QubiCConfig


# ============================================================================
# Helpers
# ============================================================================

def simple_decoder(syndrome: np.ndarray) -> np.ndarray:
    """Simple decoder: correction = syndrome (identity mapping)."""
    return syndrome.copy()


# ============================================================================
# TestULLConstants
# ============================================================================

class TestULLConstants:
    """Validate ULL timing constants and budget arithmetic."""

    def test_component_latencies_sum_within_budget(self):
        """Individual ULL component latencies must sum to <= 50ns."""
        total = (ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS +
                 ULL_TARGET_DECODE_NS + ULL_TARGET_TRIGGER_NS)
        assert total <= ULL_TARGET_TOTAL_NS

    def test_ull_budget_is_0_1_percent_of_coherence(self):
        """50ns = 0.1% of 50us coherence time."""
        assert ULL_TARGET_TOTAL_NS == 50
        coherence_us = 50.0
        budget = coherence_us * 1000 * 0.001
        assert budget == ULL_TARGET_TOTAL_NS

    def test_ultra_low_latency_mode_value(self):
        """ULTRA_LOW_LATENCY mode has value 3."""
        assert ACCLMode.ULTRA_LOW_LATENCY == 3

    def test_ull_pipeline_config_defaults(self):
        """ULLPipelineConfig defaults match constants."""
        cfg = ULLPipelineConfig()
        assert cfg.max_syndrome_bits == ULL_MAX_SYNDROME_BITS
        assert cfg.lut_depth == ULL_LUT_DECODER_DEPTH
        assert cfg.coherence_time_us == 50.0
        assert cfg.bypass_monitoring is True
        assert cfg.fiber_length_m == 1.0
        assert cfg.dma_buffer_count == ULL_DMA_BUFFER_POOL_SIZE

    def test_coherence_budget_pct_parameter(self):
        """for_qec_cycle with custom coherence_budget_pct."""
        # Default 10%
        budget_10 = LatencyBudget.for_qec_cycle(100.0, coherence_budget_pct=10.0)
        assert budget_10.total_budget_ns == 10000.0

        # Custom 1%
        budget_1 = LatencyBudget.for_qec_cycle(100.0, coherence_budget_pct=1.0)
        assert budget_1.total_budget_ns == 1000.0

        # Backward compat: default pct still 10%
        budget_default = LatencyBudget.for_qec_cycle(100.0)
        assert budget_default.total_budget_ns == budget_10.total_budget_ns


# ============================================================================
# TestDMABufferPool
# ============================================================================

class TestDMABufferPool:
    """Validate DMA buffer pool allocation and release."""

    def test_acquire_release_cycle(self):
        pool = DMABufferPool(num_buffers=4, buffer_size_bytes=64)
        assert pool.available == 4
        buf = pool.acquire()
        assert pool.available == 3
        assert pool.in_use == 1
        pool.release(buf)
        assert pool.available == 4
        assert pool.in_use == 0

    def test_exhaustion_raises(self):
        pool = DMABufferPool(num_buffers=2, buffer_size_bytes=64)
        pool.acquire()
        pool.acquire()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire()

    def test_zero_copy_identity(self):
        """get_buffer returns the same object (zero-copy proof via `is`)."""
        pool = DMABufferPool(num_buffers=4, buffer_size_bytes=64)
        buf0 = pool.get_buffer(0)
        buf0_again = pool.get_buffer(0)
        assert buf0 is buf0_again

    def test_buffer_index_out_of_range(self):
        pool = DMABufferPool(num_buffers=2, buffer_size_bytes=64)
        with pytest.raises(IndexError):
            pool.get_buffer(5)

    def test_pool_total_count(self):
        pool = DMABufferPool(num_buffers=8, buffer_size_bytes=128)
        assert pool.total == 8


# ============================================================================
# TestLUTDecoder
# ============================================================================

class TestLUTDecoder:
    """Validate LUT decoder programming and lookup."""

    def test_program_returns_entry_count(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        entries = decoder.program(simple_decoder)
        assert entries > 0
        assert decoder.programmed is True

    def test_lookup_weight1_syndrome(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        decoder.program(simple_decoder)
        # Weight-1 syndrome at bit 0
        syndrome = np.zeros(8, dtype=np.uint8)
        syndrome[0] = 1
        correction = decoder.lookup(syndrome)
        assert correction is not None
        np.testing.assert_array_equal(correction, syndrome)

    def test_bram_image_created(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        decoder.program(simple_decoder)
        image = decoder.get_bram_image()
        assert image is not None
        assert image.shape[0] == ULL_LUT_DECODER_DEPTH

    def test_depth_limit(self):
        """Entries capped at lut_depth."""
        decoder = LUTDecoder(num_syndrome_bits=16, lut_depth=10)
        entries = decoder.program(simple_decoder)
        assert entries <= 10
        assert decoder.num_entries <= 10

    def test_syndrome_bits_validation(self):
        """Syndrome bits > ULL_MAX_SYNDROME_BITS raises ValueError."""
        with pytest.raises(ValueError, match="exceeds ULL max"):
            LUTDecoder(num_syndrome_bits=ULL_MAX_SYNDROME_BITS + 1)


# ============================================================================
# TestFPGARegisterInterface
# ============================================================================

class TestFPGARegisterInterface:
    """Validate FPGA register read/write and arm/disarm."""

    def test_arm_disarm(self):
        regs = FPGARegisterInterface()
        assert not regs.is_pipeline_active()
        regs.arm_ull_pipeline()
        assert regs.is_pipeline_active()
        regs.disarm_ull_pipeline()
        assert not regs.is_pipeline_active()

    def test_read_write(self):
        regs = FPGARegisterInterface()
        regs.write(ULLRegister.SYNDROME_MASK, 0xFFFF)
        assert regs.read(ULLRegister.SYNDROME_MASK) == 0xFFFF

    def test_latency_counter(self):
        regs = FPGARegisterInterface()
        regs.set_latency_cycles(25)
        assert regs.get_last_latency_cycles() == 25

    def test_initial_state(self):
        regs = FPGARegisterInterface()
        assert regs.read(ULLRegister.ULL_CONTROL) == 0
        assert regs.read(ULLRegister.LATENCY_COUNTER) == 0


# ============================================================================
# TestHardwareAccelerator
# ============================================================================

class TestHardwareAccelerator:
    """Validate top-level hardware accelerator coordination."""

    def test_program_pipeline(self):
        accel = HardwareAccelerator()
        entries = accel.program_pipeline(simple_decoder)
        assert entries > 0
        assert accel.is_programmed is True
        assert accel.registers.is_pipeline_active() is True

    def test_estimate_latency_within_budget(self):
        """Default config should estimate < 50ns."""
        accel = HardwareAccelerator()
        latency = accel.estimate_latency_ns()
        assert latency <= ULL_TARGET_TOTAL_NS

    def test_validate_config_no_warnings_default(self):
        """Default config should pass validation."""
        accel = HardwareAccelerator()
        warnings = accel.validate_config()
        assert len(warnings) == 0

    def test_validate_config_long_fiber_warning(self):
        """Long fiber should generate a warning."""
        cfg = ULLPipelineConfig(fiber_length_m=10.0)
        accel = HardwareAccelerator(cfg)
        warnings = accel.validate_config()
        assert any("fiber" in w.lower() or "Fiber" in w for w in warnings)

    def test_disarm(self):
        accel = HardwareAccelerator()
        accel.program_pipeline(simple_decoder)
        assert accel.registers.is_pipeline_active()
        accel.disarm()
        assert not accel.registers.is_pipeline_active()


# ============================================================================
# TestDriverULLMode
# ============================================================================

class TestDriverULLMode:
    """Validate driver behavior in ULTRA_LOW_LATENCY mode."""

    def test_configure_ull_mode(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        assert accl._mode == ACCLMode.ULTRA_LOW_LATENCY
        assert accl._hw_accel is not None
        assert accl._ull_config is not None

    def test_broadcast_zero_copy(self):
        """ULL broadcast returns the same array object (zero-copy proof)."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        assert result.success
        assert result.data is data  # Zero-copy: same object

    def test_reduce_zero_copy(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = accl.reduce(data, op=ReduceOp.XOR, root=0)
        assert result.success
        assert result.data is data  # Zero-copy

    def test_allreduce_zero_copy(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = accl.allreduce(data, op=ReduceOp.XOR)
        assert result.success
        assert result.data is data  # Zero-copy

    def test_syndrome_size_limit(self):
        """Data exceeding ULL_MAX_SYNDROME_BITS returns BUFFER_ERROR."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        # Create data larger than 512 bits (64 bytes)
        large_data = np.zeros(128, dtype=np.uint8)  # 1024 bits
        result = accl.reduce(large_data, op=ReduceOp.XOR, root=0)
        assert result.status == OperationStatus.BUFFER_ERROR

    def test_mode_isolation(self):
        """Standard mode still copies data (not zero-copy)."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        assert result.success
        assert result.data is not data  # Standard mode copies

    def test_ull_broadcast_latency(self):
        """ULL broadcast reports simulated multicast latency."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        assert result.latency_ns == ULL_TARGET_MULTICAST_NS

    def test_ull_allreduce_latency(self):
        """ULL allreduce reports combined multicast + reduce latency."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1], dtype=np.uint8)
        result = accl.allreduce(data, op=ReduceOp.XOR)
        expected = ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS
        assert result.latency_ns == expected


# ============================================================================
# TestHardwareFeedbackEngine
# ============================================================================

class TestHardwareFeedbackEngine:
    """Validate hardware-autonomous feedback engine."""

    def test_program_and_cycle(self):
        engine = HardwareFeedbackEngine()
        entries = engine.program_pipeline(simple_decoder, syndrome_bits=8)
        assert entries > 0
        assert engine.is_programmed
        assert engine.is_armed

        result = engine.run_autonomous_cycle()
        assert result.success
        assert result.total_latency_ns > 0

    def test_cycle_within_budget(self):
        """Autonomous cycle latency must be <= 50ns."""
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)
        result = engine.run_autonomous_cycle()
        assert result.within_budget
        assert result.total_latency_ns <= ULL_TARGET_TOTAL_NS

    def test_phase_breakdown(self):
        """Result must include all 5 pipeline phases."""
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)
        result = engine.run_autonomous_cycle()
        assert 'readout' in result.phases
        assert 'multicast' in result.phases
        assert 'reduce' in result.phases
        assert 'decode' in result.phases
        assert 'trigger' in result.phases
        assert len(result.phases) == 5

    def test_syndrome_lookup(self):
        """Passing a syndrome triggers LUT lookup."""
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)
        syndrome = np.zeros(8, dtype=np.uint8)
        syndrome[0] = 1
        result = engine.run_autonomous_cycle(syndrome=syndrome)
        assert result.success
        assert result.correction is not None

    def test_unprogrammed_fails(self):
        engine = HardwareFeedbackEngine()
        result = engine.run_autonomous_cycle()
        assert not result.success

    def test_continuous_cycles(self):
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)
        results = engine.run_continuous(num_cycles=10)
        assert len(results) == 10
        assert all(r.success for r in results)

    def test_stats_tracking(self):
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)
        engine.run_autonomous_cycle()
        engine.run_autonomous_cycle()
        stats = engine.get_stats()
        assert stats['execution_count'] == 2
        assert stats['mean_latency_ns'] > 0
        assert stats['violations'] == 0
        assert stats['armed'] is True


# ============================================================================
# TestULLLatencyBudget
# ============================================================================

class TestULLLatencyBudget:
    """Validate ULL-specific LatencyBudget calculations."""

    def test_for_ull_feedback_50us(self):
        """for_ull_feedback(50.0) should give 50ns budget."""
        budget = LatencyBudget.for_ull_feedback(50.0)
        assert budget.total_budget_ns == 50.0

    def test_for_ull_feedback_100us(self):
        """for_ull_feedback(100.0) should give 100ns budget."""
        budget = LatencyBudget.for_ull_feedback(100.0)
        assert budget.total_budget_ns == 100.0

    def test_budget_components_sum_to_total(self):
        budget = LatencyBudget.for_ull_feedback(50.0)
        component_sum = (budget.communication_budget_ns +
                        budget.computation_budget_ns +
                        budget.margin_ns)
        assert abs(component_sum - budget.total_budget_ns) < 0.01


# ============================================================================
# TestULLEndToEnd
# ============================================================================

class TestULLEndToEnd:
    """End-to-end tests combining multiple ULL components."""

    def test_surface_code_feedback_loop(self):
        """Simulate a complete surface code QEC feedback cycle in ULL mode."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)

        # 1. Local syndrome measurement
        syndrome = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)

        # 2. Aggregate via ULL allreduce (zero-copy)
        result = accl.allreduce(syndrome, op=ReduceOp.XOR)
        assert result.success
        assert result.data is syndrome  # Zero-copy

        # 3. Verify combined latency within budget
        assert result.latency_ns <= ULL_TARGET_TOTAL_NS

    def test_multi_round_qec(self):
        """Multiple QEC rounds remain within budget."""
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(simple_decoder, syndrome_bits=8)

        for _ in range(100):
            result = engine.run_autonomous_cycle()
            assert result.within_budget

        stats = engine.get_stats()
        assert stats['violations'] == 0
        assert stats['execution_count'] == 100

    def test_mode_switching(self):
        """Switch from standard to ULL mode and back."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)

        # Standard mode: copies data
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        data = np.array([1, 0], dtype=np.uint8)
        r1 = accl.broadcast(data, root=0)
        assert r1.data is not data

        # ULL mode: zero-copy
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        r2 = accl.broadcast(data, root=0)
        assert r2.data is data

        # Back to standard
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        r3 = accl.broadcast(data, root=0)
        assert r3.data is not data

    def test_profiler_ull_phases(self):
        """Profiler recognizes ULL phase definitions."""
        profiler = CriticalPathProfiler()
        assert 'ull_feedback' in profiler._operation_phases
        assert 'ull_broadcast' in profiler._operation_phases
        assert 'ull_reduce' in profiler._operation_phases
        phases = profiler._operation_phases['ull_feedback']
        assert phases == ['readout', 'multicast', 'reduce', 'decode', 'trigger']
