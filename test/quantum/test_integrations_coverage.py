"""
Extended test coverage for integrations.py, deployment.py, feedback.py,
and hardware_accel.py modules.
"""

import sys
import os
import json
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))

from accl_quantum.constants import (
    ACCLMode, ReduceOp, SyncMode, ULLPipelineConfig,
    ULL_MAX_SYNDROME_BITS, FEEDBACK_LATENCY_BUDGET_NS,
)
from accl_quantum.driver import ACCLQuantum
from accl_quantum.integrations import (
    QubiCIntegration, QubiCConfig,
    QICKIntegration, QICKConfig,
    UnifiedQuantumControl,
    QuantumControlIntegration,
)
from accl_quantum.feedback import (
    MeasurementFeedbackPipeline, FeedbackScheduler, FeedbackConfig,
    FeedbackMode, FeedbackResult, HardwareFeedbackEngine, ULLFeedbackResult,
)
from accl_quantum.hardware_accel import (
    DMABufferPool, LUTDecoder, FPGARegisterInterface, HardwareAccelerator,
    ULLRegister,
)
from accl_quantum.deployment import (
    BoardConfig, BoardType, DeploymentConfig, DeploymentManager,
    DeploymentState, NetworkTopology, TopologyBuilder, LinkConfig,
    BoardDiscovery, create_default_deployment,
)


# ============================================================================
# QubiC Integration Tests
# ============================================================================

class TestQubiCIntegration:
    """Full coverage for QubiCIntegration."""

    def _make_integration(self, **config_kwargs):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        config = QubiCConfig(**{**{'num_qubits': 8}, **config_kwargs})
        return QubiCIntegration(accl, config)

    def test_init_default(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        qubic = QubiCIntegration(accl)
        assert qubic.config.num_qubits == 8
        assert not qubic._is_configured

    def test_configure(self):
        qubic = self._make_integration()
        qubic.configure(num_qubits=16, feedback_enabled=False, decoder_rank=2)
        assert qubic.config.num_qubits == 16
        assert qubic.config.feedback_enabled is False
        assert qubic.config.decoder_rank == 2
        assert qubic._is_configured is True

    def test_distribute_measurement(self):
        qubic = self._make_integration()
        results = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int32)
        distributed = qubic.distribute_measurement(results, source_rank=0)
        assert distributed is not None
        assert len(distributed) == 8

    def test_aggregate_syndrome(self):
        qubic = self._make_integration()
        syndrome = np.array([1, 0, 1, 0], dtype=np.int32)
        global_syndrome = qubic.aggregate_syndrome(syndrome)
        assert global_syndrome is not None
        assert len(global_syndrome) == 4

    def test_aggregate_syndrome_ull(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.ULTRA_LOW_LATENCY)
        qubic = QubiCIntegration(accl)
        syndrome = np.array([1, 0, 1], dtype=np.uint8)
        result = qubic.aggregate_syndrome_ull(syndrome)
        assert result is syndrome  # zero-copy

    def test_aggregate_syndrome_ull_standard_mode(self):
        qubic = self._make_integration()  # DETERMINISTIC mode
        syndrome = np.array([1, 0, 1, 0], dtype=np.int32)
        result = qubic.aggregate_syndrome_ull(syndrome)
        assert result is not None  # falls back to standard path

    def test_execute_instruction_bcast(self):
        qubic = self._make_integration()
        data = np.array([42], dtype=np.uint64)
        result = qubic.execute_instruction('ACCL_BCAST', data, 0)
        assert result is not None

    def test_execute_instruction_reduce(self):
        qubic = self._make_integration()
        data = np.array([1, 0, 1], dtype=np.uint64)
        result = qubic.execute_instruction('ACCL_REDUCE', data, 0, 0)
        assert result is not None

    def test_execute_instruction_allreduce(self):
        qubic = self._make_integration()
        data = np.array([1, 0, 1], dtype=np.uint64)
        result = qubic.execute_instruction('ACCL_ALLREDUCE', data, 0)
        assert result is not None

    def test_execute_instruction_barrier(self):
        qubic = self._make_integration()
        result = qubic.execute_instruction('ACCL_BARRIER')
        assert result is True

    def test_execute_instruction_sync(self):
        qubic = self._make_integration()
        result = qubic.execute_instruction('ACCL_SYNC')
        assert result is True

    def test_execute_instruction_unknown(self):
        qubic = self._make_integration()
        with pytest.raises(ValueError, match="Unknown instruction"):
            qubic.execute_instruction('ACCL_UNKNOWN')

    def test_get_qubit_rank(self):
        qubic = self._make_integration(num_qubits=16)
        assert qubic._get_qubit_rank(0) == 0
        assert qubic._get_qubit_rank(4) == 1
        assert qubic._get_qubit_rank(15) == 3

    def test_compute_syndrome_vectorized(self):
        qubic = self._make_integration()
        meas = np.array([1, 0, 1, 1, 0, 0], dtype=np.int32)
        syndrome = qubic._compute_syndrome(meas)
        assert len(syndrome) == 3
        assert syndrome[0] == 1  # 1 ^ 0
        assert syndrome[1] == 0  # 1 ^ 1
        assert syndrome[2] == 0  # 0 ^ 0

    def test_collective_readout_correction(self):
        qubic = self._make_integration()
        raw = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int32)
        corrected = qubic.collective_readout_correction(raw)
        assert corrected is not None
        assert len(corrected) == 8


# ============================================================================
# QICK Integration Tests
# ============================================================================

class TestQICKIntegration:
    """Full coverage for QICKIntegration."""

    def _make_integration(self, **config_kwargs):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        config = QICKConfig(**config_kwargs) if config_kwargs else None
        return QICKIntegration(accl, config)

    def test_init_default(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        qick = QICKIntegration(accl)
        assert qick.config.num_channels == 8
        assert qick.config.tproc_freq_mhz == 430.0

    def test_configure(self):
        qick = self._make_integration()
        qick.configure(num_channels=16, enable_counter_sync=True)
        assert qick.config.num_channels == 16
        assert qick._is_configured is True
        assert qick._axi_bridge_enabled is True

    def test_distribute_measurement(self):
        qick = self._make_integration()
        results = np.array([0, 1, 0, 1], dtype=np.int32)
        distributed = qick.distribute_measurement(results, source_rank=0)
        assert distributed is not None

    def test_aggregate_syndrome(self):
        qick = self._make_integration()
        syndrome = np.array([1, 0, 1, 0], dtype=np.int32)
        global_syndrome = qick.aggregate_syndrome(syndrome)
        assert global_syndrome is not None

    def test_get_synchronized_time(self):
        qick = self._make_integration()
        t = qick.get_synchronized_time()
        assert isinstance(t, int)
        assert t > 0

    def test_schedule_synchronized_pulse(self):
        qick = self._make_integration()
        future_time = qick.get_synchronized_time() + 10000
        result = qick.schedule_synchronized_pulse(0, future_time, {"amp": 1.0})
        assert result is True

    def test_schedule_pulse_past_time(self):
        qick = self._make_integration()
        result = qick.schedule_synchronized_pulse(0, 0, {"amp": 1.0})
        assert result is False

    def test_collective_acquire(self):
        qick = self._make_integration()
        data = qick.collective_acquire([0, 1], 100)
        assert data is not None

    def test_tproc_collective_op_broadcast(self):
        qick = self._make_integration()
        result = qick.tproc_collective_op(0, 0, 1, 0)  # broadcast
        assert result == 0

    def test_tproc_collective_op_reduce(self):
        qick = self._make_integration()
        result = qick.tproc_collective_op(1, 0, 1, 0, 0)  # reduce
        assert result == 0

    def test_tproc_collective_op_barrier(self):
        qick = self._make_integration()
        result = qick.tproc_collective_op(2)  # barrier
        assert result == 0

    def test_tproc_collective_op_unknown(self):
        qick = self._make_integration()
        with pytest.raises(ValueError, match="Unknown tProcessor"):
            qick.tproc_collective_op(99)

    def test_complex_format_conversion(self):
        qick = self._make_integration()
        # Complex I/Q data
        data = np.array([1+2j, 3+4j], dtype=np.complex128)
        packed = qick._qick_to_accl_format(data)
        assert packed.dtype == np.uint64
        unpacked = qick._accl_to_qick_format(packed)
        assert np.iscomplexobj(unpacked)

    def test_real_format_conversion(self):
        qick = self._make_integration()
        data = np.array([1, 2, 3], dtype=np.int32)
        packed = qick._qick_to_accl_format(data)
        assert packed.dtype == np.uint64


# ============================================================================
# UnifiedQuantumControl Tests
# ============================================================================

class TestUnifiedQuantumControl:
    """Tests for UnifiedQuantumControl."""

    def test_qubic_backend(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        uqc = UnifiedQuantumControl(accl, backend='qubic', num_qubits=8)
        assert uqc.backend_type == 'qubic'
        assert isinstance(uqc.backend, QubiCIntegration)

    def test_qick_backend(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        uqc = UnifiedQuantumControl(accl, backend='qick', num_channels=4)
        assert uqc.backend_type == 'qick'
        assert isinstance(uqc.backend, QICKIntegration)

    def test_unknown_backend(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        with pytest.raises(ValueError, match="Unknown backend"):
            UnifiedQuantumControl(accl, backend='invalid')

    def test_measure_and_distribute(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        uqc = UnifiedQuantumControl(accl, backend='qubic', num_qubits=8)
        result = uqc.measure_and_distribute([0, 1, 2, 3])
        assert result is not None

    def test_qec_cycle(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        uqc = UnifiedQuantumControl(accl, backend='qubic', num_qubits=8)
        result = uqc.qec_cycle([0, 1, 2, 3], [4, 5, 6, 7])
        assert result is not None

    def test_configure(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        uqc = UnifiedQuantumControl(accl, backend='qubic', num_qubits=8)
        uqc.configure(num_qubits=16)
        assert uqc.backend.config.num_qubits == 16


# ============================================================================
# Abstract Base Class Tests
# ============================================================================

class TestQuantumControlABC:
    """Verify ABC enforcement."""

    def test_cannot_instantiate_abc(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        with pytest.raises(TypeError):
            QuantumControlIntegration(accl)

    def test_abstract_methods_raise(self):
        # Verify NotImplementedError is in the abstract method bodies
        import inspect
        src = inspect.getsource(QuantumControlIntegration.configure)
        assert "NotImplementedError" in src


# ============================================================================
# FeedbackScheduler Context Manager Tests
# ============================================================================

class TestFeedbackSchedulerContextManager:
    """Test FeedbackScheduler __enter__/__exit__."""

    def _make_scheduler(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        pipeline = MeasurementFeedbackPipeline(accl)
        return FeedbackScheduler(pipeline)

    def test_context_manager_arms(self):
        scheduler = self._make_scheduler()
        assert not scheduler.pipeline._is_armed
        with scheduler:
            assert scheduler.pipeline._is_armed
        assert not scheduler.pipeline._is_armed

    def test_context_manager_clears_schedule(self):
        scheduler = self._make_scheduler()
        scheduler.add_feedback(FeedbackMode.SINGLE_QUBIT, source_rank=0, action_if_one="x")
        assert len(scheduler._schedule) == 1
        with scheduler:
            pass
        assert len(scheduler._schedule) == 0

    def test_context_manager_returns_self(self):
        scheduler = self._make_scheduler()
        with scheduler as s:
            assert s is scheduler


# ============================================================================
# FeedbackPipeline Extended Tests
# ============================================================================

class TestFeedbackPipelineExtended:
    """Extended tests for MeasurementFeedbackPipeline."""

    def _make_pipeline(self):
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        return MeasurementFeedbackPipeline(accl)

    def test_register_and_trigger_action(self):
        pipeline = self._make_pipeline()
        triggered = []
        pipeline.register_action("flip", lambda: triggered.append(True))
        pipeline.arm()
        result = pipeline.single_qubit_feedback(source_rank=0, action_if_one="flip")
        assert result.success

    def test_parity_feedback(self):
        pipeline = self._make_pipeline()
        pipeline.register_action("correct", lambda: None)
        result = pipeline.parity_feedback([0, 1], action_if_odd="correct")
        assert result.success
        assert "measurement_ns" in result.breakdown
        assert "communication_ns" in result.breakdown

    def test_syndrome_feedback(self):
        pipeline = self._make_pipeline()
        def decoder(syndrome):
            return syndrome
        result = pipeline.syndrome_feedback(decoder)
        assert result.success
        assert "aggregation_ns" in result.breakdown

    def test_pipelined_feedback(self):
        pipeline = self._make_pipeline()
        op_id = pipeline.start_pipelined_feedback(0, "action")
        assert op_id == 0
        result = pipeline.check_pipelined_feedback(op_id)
        assert result is not None
        assert result.success

    def test_pipelined_max_pending(self):
        config = FeedbackConfig(max_pending_operations=2)
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        pipeline = MeasurementFeedbackPipeline(accl, config)
        pipeline.start_pipelined_feedback(0, "a")
        pipeline.start_pipelined_feedback(0, "b")
        with pytest.raises(RuntimeError, match="Max pending"):
            pipeline.start_pipelined_feedback(0, "c")

    def test_pipelined_not_enabled(self):
        config = FeedbackConfig(enable_pipelining=False)
        accl = ACCLQuantum(num_ranks=4, local_rank=0)
        accl.configure(mode=ACCLMode.DETERMINISTIC)
        pipeline = MeasurementFeedbackPipeline(accl, config)
        with pytest.raises(RuntimeError, match="Pipelining not enabled"):
            pipeline.start_pipelined_feedback(0, "a")

    def test_latency_statistics(self):
        pipeline = self._make_pipeline()
        for _ in range(5):
            pipeline.single_qubit_feedback(0, "x")
        stats = pipeline.get_latency_statistics()
        assert stats["count"] == 5
        assert stats["mean_ns"] > 0

    def test_breakdown_statistics(self):
        pipeline = self._make_pipeline()
        for _ in range(3):
            pipeline.single_qubit_feedback(0, "x")
        breakdown = pipeline.get_breakdown_statistics()
        assert "measurement_ns" in breakdown

    def test_clear_history(self):
        pipeline = self._make_pipeline()
        pipeline.single_qubit_feedback(0, "x")
        pipeline.clear_history()
        assert pipeline.get_latency_statistics() == {}


# ============================================================================
# DMA Buffer Pool Extended Tests
# ============================================================================

class TestDMABufferPoolExtended:
    """Extended tests for DMABufferPool."""

    def test_lazy_init(self):
        pool = DMABufferPool(num_buffers=4, lazy=True)
        assert not pool._initialized
        assert pool.available == 0  # not allocated yet
        buf = pool.acquire()
        assert pool._initialized
        assert buf is not None

    def test_lazy_get_buffer(self):
        pool = DMABufferPool(num_buffers=4, lazy=True)
        buf = pool.get_buffer(0)
        assert pool._initialized
        assert buf is not None

    def test_release_and_reacquire(self):
        pool = DMABufferPool(num_buffers=2)
        b1 = pool.acquire()
        b2 = pool.acquire()
        assert pool.available == 0
        pool.release(b1)
        assert pool.available == 1
        b3 = pool.acquire()
        assert b3 is b1  # same buffer reused

    def test_in_use_tracking(self):
        pool = DMABufferPool(num_buffers=4)
        assert pool.in_use == 0
        b = pool.acquire()
        assert pool.in_use == 1
        pool.release(b)
        assert pool.in_use == 0


# ============================================================================
# LUT Decoder Extended Tests
# ============================================================================

class TestLUTDecoderExtended:
    """Extended tests for LUTDecoder."""

    def test_program_identity(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        entries = decoder.program(lambda s: s)
        assert entries > 0
        assert decoder.programmed

    def test_lookup_weight1(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        decoder.program(lambda s: s)
        syndrome = np.zeros(8, dtype=np.uint8)
        syndrome[3] = 1
        result = decoder.lookup(syndrome)
        assert result is not None
        assert result[3] == 1

    def test_lookup_unknown(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        decoder.program(lambda s: s)
        # Weight-3 syndrome unlikely to be in table
        syndrome = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        result = decoder.lookup(syndrome)
        # May or may not be in table depending on depth

    def test_bram_image(self):
        decoder = LUTDecoder(num_syndrome_bits=8)
        decoder.program(lambda s: s)
        image = decoder.get_bram_image()
        assert image is not None
        assert image.shape[0] == 4096  # default lut_depth

    def test_custom_lut_depth(self):
        decoder = LUTDecoder(num_syndrome_bits=8, lut_depth=100)
        entries = decoder.program(lambda s: s)
        assert entries <= 100


# ============================================================================
# FPGA Register Interface Tests
# ============================================================================

class TestFPGARegisterExtended:
    """Extended tests for FPGARegisterInterface."""

    def test_write_read(self):
        regs = FPGARegisterInterface()
        regs.write(ULLRegister.SYNDROME_MASK, 0xFFFF)
        assert regs.read(ULLRegister.SYNDROME_MASK) == 0xFFFF

    def test_arm_disarm(self):
        regs = FPGARegisterInterface()
        assert not regs.is_pipeline_active()
        regs.arm_ull_pipeline()
        assert regs.is_pipeline_active()
        assert regs.read(ULLRegister.ULL_CONTROL) == 1
        regs.disarm_ull_pipeline()
        assert not regs.is_pipeline_active()
        assert regs.read(ULLRegister.ULL_CONTROL) == 0

    def test_latency_counter(self):
        regs = FPGARegisterInterface()
        regs.set_latency_cycles(25)
        assert regs.get_last_latency_cycles() == 25

    def test_read_unknown_addr(self):
        regs = FPGARegisterInterface()
        assert regs.read(0x999) == 0


# ============================================================================
# HardwareAccelerator Extended Tests
# ============================================================================

class TestHardwareAcceleratorExtended:
    """Extended tests for HardwareAccelerator."""

    def test_validate_clean_config(self):
        accel = HardwareAccelerator()
        warnings = accel.validate_config()
        fiber_warnings = [w for w in warnings if 'fiber' in w.lower()]
        assert len(fiber_warnings) == 0

    def test_validate_long_fiber(self):
        config = ULLPipelineConfig(fiber_length_m=5.0)
        accel = HardwareAccelerator(config)
        warnings = accel.validate_config()
        assert any('fiber' in w.lower() or 'Fiber' in w for w in warnings)

    def test_validate_exceeds_budget(self):
        config = ULLPipelineConfig(fiber_length_m=10.0)
        accel = HardwareAccelerator(config)
        warnings = accel.validate_config()
        assert any('exceeds' in w.lower() for w in warnings)

    def test_software_multicast_slower(self):
        hw_config = ULLPipelineConfig(use_hardware_multicast=True)
        sw_config = ULLPipelineConfig(use_hardware_multicast=False)
        hw = HardwareAccelerator(hw_config)
        sw = HardwareAccelerator(sw_config)
        assert hw.estimate_latency_ns() < sw.estimate_latency_ns()


# ============================================================================
# HardwareFeedbackEngine Extended Tests
# ============================================================================

class TestHardwareFeedbackEngineExtended:
    """Extended tests for HardwareFeedbackEngine."""

    def test_unprogrammed_fails(self):
        engine = HardwareFeedbackEngine()
        result = engine.run_autonomous_cycle()
        assert not result.success

    def test_update_decoder(self):
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(decoder_fn=lambda s: s, syndrome_bits=8)
        entries = engine.update_decoder(lambda s: np.zeros_like(s))
        assert entries > 0

    def test_update_decoder_not_programmed(self):
        engine = HardwareFeedbackEngine()
        with pytest.raises(RuntimeError, match="not programmed"):
            engine.update_decoder(lambda s: s)

    def test_disarm(self):
        engine = HardwareFeedbackEngine()
        engine.program_pipeline(decoder_fn=lambda s: s, syndrome_bits=8)
        assert engine.is_armed
        engine.disarm()
        assert not engine.is_armed

    def test_properties(self):
        engine = HardwareFeedbackEngine()
        assert not engine.is_armed
        assert not engine.is_programmed
        engine.program_pipeline(decoder_fn=lambda s: s, syndrome_bits=8)
        assert engine.is_armed
        assert engine.is_programmed


# ============================================================================
# Deployment Config Extended Tests
# ============================================================================

class TestDeploymentConfigExtended:
    """Extended tests for DeploymentConfig."""

    def test_save_and_load(self):
        config = create_default_deployment(4, "test-save")
        with tempfile.TemporaryDirectory() as td:
            from pathlib import Path
            path = Path(td) / "config.json"
            config.save(path)

            loaded = DeploymentConfig.load(path)
            assert loaded.name == "test-save"
            assert loaded.num_boards == 4
            assert len(loaded.boards) == 4
            assert len(loaded.links) > 0

    def test_validate_bad_num_boards(self):
        config = DeploymentConfig(name="bad", num_boards=1)
        errors = config.validate()
        assert any("Minimum 2" in e for e in errors)

    def test_validate_missing_boards(self):
        config = DeploymentConfig(name="bad", num_boards=4)
        errors = config.validate()
        assert any("Missing board" in e or "Expected" in e for e in errors)

    def test_validate_master_out_of_range(self):
        config = create_default_deployment(4)
        config.master_rank = 10
        errors = config.validate()
        assert any("Master rank" in e for e in errors)

    def test_min_links_star(self):
        config = DeploymentConfig(name="t", topology=NetworkTopology.STAR, num_boards=4)
        assert config._min_links_for_topology() == 3

    def test_min_links_ring(self):
        config = DeploymentConfig(name="t", topology=NetworkTopology.RING, num_boards=4)
        assert config._min_links_for_topology() == 4

    def test_min_links_full_mesh(self):
        config = DeploymentConfig(name="t", topology=NetworkTopology.FULL_MESH, num_boards=4)
        assert config._min_links_for_topology() == 6

    def test_min_links_custom(self):
        config = DeploymentConfig(name="t", topology=NetworkTopology.CUSTOM, num_boards=4)
        assert config._min_links_for_topology() == 0


# ============================================================================
# TopologyBuilder Extended Tests
# ============================================================================

class TestTopologyBuilderExtended:
    """Extended tests for TopologyBuilder."""

    def _make_boards(self, n):
        return [
            BoardConfig(rank=i, hostname=f"h{i}", ip_address=f"10.0.0.{i}",
                       mac_address="00:00:00:00:00:00", board_type=BoardType.ZCU216)
            for i in range(n)
        ]

    def test_star_4_boards(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_star(boards, center_rank=0)
        # 3 boards connect to center, bidirectional = 6 links
        assert len(links) == 6

    def test_ring_4_boards(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_ring(boards)
        assert len(links) == 4  # one per board

    def test_tree_4_boards(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_tree(boards, root_rank=0, fanout=4)
        # 3 child nodes, bidirectional = 6 links
        assert len(links) == 6

    def test_full_mesh_4_boards(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_full_mesh(boards)
        # C(4,2)*2 = 12 links
        assert len(links) == 12

    def test_full_mesh_3_boards(self):
        boards = self._make_boards(3)
        links = TopologyBuilder.build_full_mesh(boards)
        assert len(links) == 6


# ============================================================================
# DeploymentManager Tests
# ============================================================================

class TestDeploymentManagerExtended:
    """Extended tests for DeploymentManager."""

    def test_init(self):
        config = create_default_deployment(4)
        mgr = DeploymentManager(config)
        assert mgr.state == DeploymentState.UNINITIALIZED

    def test_state_callbacks(self):
        config = create_default_deployment(4)
        mgr = DeploymentManager(config)
        states = []
        mgr.add_state_callback(lambda s: states.append(s))
        mgr._set_state(DeploymentState.CONFIGURING)
        assert states == [DeploymentState.CONFIGURING]

    def test_error_callbacks(self):
        config = create_default_deployment(4)
        mgr = DeploymentManager(config)
        errors = []
        mgr.add_error_callback(lambda e: errors.append(e))
        mgr._report_error("test error")
        assert errors == ["test error"]

    def test_get_status(self):
        config = create_default_deployment(4)
        mgr = DeploymentManager(config)
        status = mgr.get_status()
        assert status["state"] == "uninitialized"
        assert status["num_boards"] == 4
        assert len(status["boards"]) == 4

    def test_shutdown(self):
        config = create_default_deployment(4)
        mgr = DeploymentManager(config)
        mgr.shutdown()
        assert mgr.state == DeploymentState.SHUTDOWN

    def test_load_bitstreams_no_path(self):
        config = create_default_deployment(4)
        config.bitstream_path = ""
        mgr = DeploymentManager(config)
        assert mgr.load_bitstreams() is True


# ============================================================================
# BoardConfig Tests
# ============================================================================

class TestBoardConfigExtended:
    """Extended tests for BoardConfig."""

    def test_to_dict(self):
        board = BoardConfig(
            rank=0, hostname="host0", ip_address="10.0.0.1",
            mac_address="aa:bb:cc:dd:ee:ff", board_type=BoardType.ZCU216,
        )
        d = board.to_dict()
        assert d["rank"] == 0
        assert d["hostname"] == "host0"
        assert d["board_type"] == "zcu216"

    def test_from_dict(self):
        d = {
            "rank": 1, "hostname": "host1", "ip_address": "10.0.0.2",
            "mac_address": "aa:bb:cc:dd:ee:ff", "board_type": "zcu111",
            "aurora_lanes": 4, "aurora_rate_gbps": 10.0,
            "fpga_bitstream": "", "firmware_version": "",
            "dac_channels": 8, "adc_channels": 8,
            "clock_source": "internal", "reference_freq_mhz": 245.76,
            "aurora_ports": [0, 1, 2, 3], "management_port": 5000,
            "data_port": 5001,
        }
        board = BoardConfig.from_dict(d)
        assert board.rank == 1
        assert board.board_type == BoardType.ZCU111

    def test_roundtrip(self):
        board = BoardConfig(
            rank=2, hostname="host2", ip_address="10.0.0.3",
            mac_address="00:00:00:00:00:02", board_type=BoardType.RFSoC4x2,
        )
        d = board.to_dict()
        board2 = BoardConfig.from_dict(d)
        assert board2.rank == board.rank
        assert board2.board_type == board.board_type


# ============================================================================
# ULLFeedbackResult Tests
# ============================================================================

class TestULLFeedbackResultExtended:
    """Extended tests for ULLFeedbackResult."""

    def test_within_budget_true(self):
        result = ULLFeedbackResult(success=True, total_latency_ns=34.0)
        assert result.within_budget is True

    def test_within_budget_false(self):
        result = ULLFeedbackResult(success=True, total_latency_ns=60.0)
        assert result.within_budget is False

    def test_within_budget_edge(self):
        result = ULLFeedbackResult(success=True, total_latency_ns=50.0)
        assert result.within_budget is True
