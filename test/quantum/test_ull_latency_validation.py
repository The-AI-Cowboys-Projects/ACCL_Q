"""
ULL-specific latency validation tests.

Validates that the ULL pipeline meets the 50ns budget target and that
component latencies are consistent with the HLS cycle counts.
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))

from accl_quantum.constants import (
    ULL_TARGET_MULTICAST_NS, ULL_TARGET_REDUCE_NS,
    ULL_TARGET_DECODE_NS, ULL_TARGET_TRIGGER_NS,
    ULL_TARGET_TOTAL_NS, ULL_MAX_JITTER_NS,
    ULL_MAX_SYNDROME_BITS, ULL_LUT_DECODER_DEPTH,
    CLOCK_PERIOD_NS, FIBER_DELAY_NS_PER_METER,
    ULLPipelineConfig, LatencyBudget, ACCLMode,
)
from accl_quantum.hardware_accel import (
    HardwareAccelerator, DMABufferPool, LUTDecoder, FPGARegisterInterface,
    ULLRegister,
)
from accl_quantum.feedback import HardwareFeedbackEngine, ULLFeedbackResult
from accl_quantum.driver import ACCLQuantum


class TestULLBudgetArithmetic:
    """Validate that ULL budget arithmetic is internally consistent."""

    def test_component_sum_within_budget(self):
        """Sum of component targets must be <= total budget."""
        total = (ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS +
                 ULL_TARGET_DECODE_NS + ULL_TARGET_TRIGGER_NS)
        assert total <= ULL_TARGET_TOTAL_NS

    def test_cycle_count_consistency(self):
        """Component targets must equal cycle_count * clock_period."""
        assert ULL_TARGET_MULTICAST_NS == 5 * CLOCK_PERIOD_NS  # 5 cycles * 2ns
        assert ULL_TARGET_REDUCE_NS == 2 * CLOCK_PERIOD_NS     # 2 cycles * 2ns
        assert ULL_TARGET_DECODE_NS == 4 * CLOCK_PERIOD_NS     # 4 cycles * 2ns
        assert ULL_TARGET_TRIGGER_NS == 1 * CLOCK_PERIOD_NS    # 1 cycle * 2ns

    def test_total_budget_equals_25_cycles(self):
        """50ns = 25 cycles at 500MHz."""
        assert ULL_TARGET_TOTAL_NS == 25 * CLOCK_PERIOD_NS

    def test_coherence_budget_derivation(self):
        """Budget should be 0.1% of coherence time."""
        budget = LatencyBudget.for_ull_feedback(coherence_time_us=50.0)
        assert budget.total_budget_ns == 50.0
        # 50ns = 0.1% of 50us (50000ns)
        assert budget.total_budget_ns / 50000 == 0.001


class TestULLHardwareAcceleratorLatency:
    """Validate HardwareAccelerator latency estimates."""

    def test_default_config_within_budget(self):
        """Default ULL config should produce latency within 50ns."""
        accel = HardwareAccelerator()
        estimated = accel.estimate_latency_ns()
        assert estimated <= ULL_TARGET_TOTAL_NS

    def test_short_fiber_within_budget(self):
        """1m fiber adds only 5ns, should be within budget."""
        config = ULLPipelineConfig(fiber_length_m=1.0)
        accel = HardwareAccelerator(config)
        estimated = accel.estimate_latency_ns()
        fiber_component = 1.0 * FIBER_DELAY_NS_PER_METER
        assert fiber_component == 5.0
        assert estimated <= ULL_TARGET_TOTAL_NS

    def test_long_fiber_exceeds_budget(self):
        """10m fiber adds 50ns — should exceed budget."""
        config = ULLPipelineConfig(fiber_length_m=10.0)
        accel = HardwareAccelerator(config)
        estimated = accel.estimate_latency_ns()
        assert estimated > ULL_TARGET_TOTAL_NS

    def test_hardware_multicast_vs_software(self):
        """Hardware multicast should be faster than software fallback."""
        hw_config = ULLPipelineConfig(use_hardware_multicast=True)
        sw_config = ULLPipelineConfig(use_hardware_multicast=False)
        hw_accel = HardwareAccelerator(hw_config)
        sw_accel = HardwareAccelerator(sw_config)
        assert hw_accel.estimate_latency_ns() < sw_accel.estimate_latency_ns()

    def test_validation_warnings_long_fiber(self):
        """Config with long fiber should produce warnings."""
        config = ULLPipelineConfig(fiber_length_m=5.0)
        accel = HardwareAccelerator(config)
        warnings = accel.validate_config()
        fiber_warnings = [w for w in warnings if 'fiber' in w.lower() or 'Fiber' in w]
        assert len(fiber_warnings) > 0

    def test_validation_clean_default(self):
        """Default config should produce no warnings."""
        accel = HardwareAccelerator()
        warnings = accel.validate_config()
        # Default has 1m fiber, should be clean
        fiber_warnings = [w for w in warnings if 'fiber' in w.lower() or 'Fiber' in w]
        assert len(fiber_warnings) == 0


class TestULLFeedbackLatency:
    """Validate HardwareFeedbackEngine produces results within budget."""

    def _make_engine(self, **config_kwargs):
        config = ULLPipelineConfig(**config_kwargs)
        engine = HardwareFeedbackEngine(config)
        engine.program_pipeline(
            decoder_fn=lambda s: s,
            syndrome_bits=16,
        )
        return engine

    def test_single_cycle_within_budget(self):
        engine = self._make_engine()
        result = engine.run_autonomous_cycle()
        assert result.success
        assert result.within_budget
        assert result.total_latency_ns <= ULL_TARGET_TOTAL_NS

    def test_phases_sum_to_total(self):
        engine = self._make_engine()
        result = engine.run_autonomous_cycle()
        phase_sum = sum(result.phases.values())
        # Allow small float rounding
        assert abs(phase_sum - result.total_latency_ns) < 1.0

    def test_continuous_no_violations(self):
        engine = self._make_engine()
        results = engine.run_continuous(num_cycles=100)
        violations = [r for r in results if not r.within_budget]
        assert len(violations) == 0

    def test_phase_names(self):
        engine = self._make_engine()
        result = engine.run_autonomous_cycle()
        expected_phases = {'readout', 'multicast', 'reduce', 'decode', 'trigger'}
        assert set(result.phases.keys()) == expected_phases

    def test_each_phase_positive(self):
        engine = self._make_engine()
        result = engine.run_autonomous_cycle()
        for phase, ns in result.phases.items():
            assert ns > 0, f"Phase {phase} has non-positive latency: {ns}"

    def test_stats_tracking(self):
        engine = self._make_engine()
        for _ in range(10):
            engine.run_autonomous_cycle()
        stats = engine.get_stats()
        assert stats['execution_count'] == 10
        assert stats['violations'] == 0


class TestULLDriverLatency:
    """Validate driver ULL mode returns correct latency values."""

    def _make_ull_driver(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        return accl

    def test_broadcast_latency(self):
        accl = self._make_ull_driver()
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        assert result.latency_ns == ULL_TARGET_MULTICAST_NS

    def test_reduce_latency(self):
        accl = self._make_ull_driver()
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        from accl_quantum.constants import ReduceOp
        result = accl.reduce(data, op=ReduceOp.XOR, root=0)
        assert result.latency_ns == ULL_TARGET_REDUCE_NS

    def test_allreduce_latency(self):
        accl = self._make_ull_driver()
        data = np.array([1, 0, 1, 0], dtype=np.uint8)
        from accl_quantum.constants import ReduceOp
        result = accl.allreduce(data, op=ReduceOp.XOR)
        expected = ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS
        assert result.latency_ns == expected


class TestULLZeroCopyProof:
    """Validate zero-copy semantics in ULL mode."""

    def test_broadcast_zero_copy(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([42, 137, 255], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        assert result.data is data  # identity check, not equality

    def test_reduce_zero_copy(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1, 0, 1], dtype=np.uint8)
        from accl_quantum.constants import ReduceOp
        result = accl.reduce(data, op=ReduceOp.XOR, root=0)
        assert result.data is data

    def test_allreduce_zero_copy(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        data = np.array([1, 0, 1], dtype=np.uint8)
        from accl_quantum.constants import ReduceOp
        result = accl.allreduce(data, op=ReduceOp.XOR)
        assert result.data is data

    def test_standard_mode_copies(self):
        """Standard mode should copy data, NOT zero-copy."""
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        data = np.array([42, 137, 255], dtype=np.uint8)
        result = accl.broadcast(data, root=0)
        # Standard mode copies
        assert result.data is not data
        assert np.array_equal(result.data, data)


class TestULLSyndromeLimits:
    """Validate syndrome size enforcement in ULL mode."""

    def test_max_syndrome_bits(self):
        """LUT decoder should reject syndrome > max bits."""
        from accl_quantum.hardware_accel import LUTDecoder
        with pytest.raises(ValueError, match="exceeds ULL max"):
            LUTDecoder(num_syndrome_bits=ULL_MAX_SYNDROME_BITS + 1)

    def test_max_syndrome_accepted(self):
        """Exactly max bits should be accepted."""
        from accl_quantum.hardware_accel import LUTDecoder
        decoder = LUTDecoder(num_syndrome_bits=ULL_MAX_SYNDROME_BITS)
        assert decoder._num_bits == ULL_MAX_SYNDROME_BITS


class TestULLModeIsolation:
    """Validate that ULL mode doesn't affect other modes."""

    def test_switch_to_ull_and_back(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)

        # Standard mode
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        data = np.array([1, 2, 3], dtype=np.uint8)
        r1 = accl.broadcast(data, root=0)
        assert r1.data is not data  # Standard mode copies

        # Switch to ULL
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        r2 = accl.broadcast(data, root=0)
        assert r2.data is data  # ULL is zero-copy

        # Switch back
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        r3 = accl.broadcast(data, root=0)
        assert r3.data is not data  # Back to standard
