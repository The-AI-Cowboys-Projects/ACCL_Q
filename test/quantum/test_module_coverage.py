"""
Tests for ACCL-Q modules: profiler, stats, deployment, constants.

Fills coverage gaps for modules that previously lacked dedicated tests.
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'driver', 'python'))

from accl_quantum.constants import (
    ACCLMode, ACCLConfig, ReduceOp, SyncMode, CollectiveOp,
    OperationStatus, QuantumMsgType, LatencyBudget, ULLPipelineConfig,
    CLOCK_PERIOD_NS, MAX_RANKS, TARGET_P2P_LATENCY_NS,
    TARGET_BROADCAST_LATENCY_NS, TARGET_REDUCE_LATENCY_NS,
    ULL_TARGET_TOTAL_NS, ULL_TARGET_MULTICAST_NS,
    ULL_TARGET_REDUCE_NS, ULL_TARGET_DECODE_NS, ULL_TARGET_TRIGGER_NS,
    ULL_MAX_SYNDROME_BITS, ULL_LUT_DECODER_DEPTH,
    FEEDBACK_LATENCY_BUDGET_NS, MAX_JITTER_NS,
)
from accl_quantum.stats import (
    LatencyStats, LatencyRecord, LatencyMonitor, LatencyProfiler,
)
from accl_quantum.profiler import (
    CriticalPathProfiler, BottleneckAnalyzer, OptimizationAdvisor,
    PerformanceRegressor, LatencyVisualizer, ProfilingSession,
    LatencyBreakdown, Bottleneck, Recommendation,
    BottleneckType, OptimizationCategory, ProfileSample,
)
from accl_quantum.deployment import (
    BoardType, NetworkTopology, DeploymentState,
    BoardConfig, LinkConfig, DeploymentConfig,
    TopologyBuilder, DeploymentManager,
    create_default_deployment,
)


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Test constants module values and consistency."""

    def test_clock_period(self):
        assert CLOCK_PERIOD_NS == 2
        assert 1000 / CLOCK_PERIOD_NS == 500  # 500 MHz

    def test_max_ranks(self):
        assert MAX_RANKS == 16

    def test_latency_targets_ordered(self):
        """P2P < Broadcast < Reduce makes physical sense."""
        assert TARGET_P2P_LATENCY_NS < TARGET_BROADCAST_LATENCY_NS
        assert TARGET_BROADCAST_LATENCY_NS <= TARGET_REDUCE_LATENCY_NS

    def test_ull_targets_sum_within_budget(self):
        component_sum = (
            ULL_TARGET_MULTICAST_NS + ULL_TARGET_REDUCE_NS +
            ULL_TARGET_DECODE_NS + ULL_TARGET_TRIGGER_NS
        )
        assert component_sum <= ULL_TARGET_TOTAL_NS

    def test_ull_constants_positive(self):
        assert ULL_MAX_SYNDROME_BITS > 0
        assert ULL_LUT_DECODER_DEPTH > 0

    def test_feedback_budget(self):
        assert FEEDBACK_LATENCY_BUDGET_NS == 500


class TestACCLMode:
    """Test ACCLMode enum."""

    def test_modes_distinct(self):
        modes = [ACCLMode.STANDARD, ACCLMode.DETERMINISTIC,
                 ACCLMode.LOW_LATENCY, ACCLMode.ULTRA_LOW_LATENCY]
        assert len(set(modes)) == 4

    def test_ull_mode_value(self):
        assert ACCLMode.ULTRA_LOW_LATENCY == 3

    def test_mode_ordering(self):
        assert ACCLMode.STANDARD < ACCLMode.ULTRA_LOW_LATENCY


class TestACCLConfig:
    """Test ACCLConfig dataclass."""

    def test_default_values(self):
        config = ACCLConfig(num_ranks=4, local_rank=0)
        assert config.mode == ACCLMode.DETERMINISTIC
        assert config.sync_mode == SyncMode.HARDWARE
        assert config.enable_latency_monitoring is True

    def test_validate_valid(self):
        config = ACCLConfig(num_ranks=8, local_rank=3)
        assert config.validate() is True

    def test_validate_invalid_ranks(self):
        config = ACCLConfig(num_ranks=0, local_rank=0)
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_invalid_local_rank(self):
        config = ACCLConfig(num_ranks=4, local_rank=5)
        with pytest.raises(ValueError):
            config.validate()

    def test_validate_max_ranks(self):
        config = ACCLConfig(num_ranks=MAX_RANKS + 1, local_rank=0)
        with pytest.raises(ValueError):
            config.validate()


class TestULLPipelineConfig:
    """Test ULLPipelineConfig defaults."""

    def test_defaults(self):
        config = ULLPipelineConfig()
        assert config.max_syndrome_bits == ULL_MAX_SYNDROME_BITS
        assert config.decoder_type == 'lut'
        assert config.coherence_time_us == 50.0
        assert config.auto_trigger is True
        assert config.bypass_monitoring is True

    def test_custom_config(self):
        config = ULLPipelineConfig(
            max_syndrome_bits=64,
            fiber_length_m=0.5,
            coherence_time_us=100.0,
        )
        assert config.max_syndrome_bits == 64
        assert config.fiber_length_m == 0.5


class TestLatencyBudget:
    """Test LatencyBudget factory methods."""

    def test_for_qec_cycle_default(self):
        budget = LatencyBudget.for_qec_cycle()
        assert budget.total_budget_ns == 10000  # 100us * 1000 * 10%
        assert budget.communication_budget_ns == 6000
        assert budget.computation_budget_ns == 3000
        assert budget.margin_ns == 1000

    def test_for_qec_cycle_custom_pct(self):
        budget = LatencyBudget.for_qec_cycle(coherence_time_us=50.0, coherence_budget_pct=1.0)
        assert budget.total_budget_ns == 500  # 50 * 1000 * 1%

    def test_for_feedback(self):
        budget = LatencyBudget.for_feedback()
        assert budget.total_budget_ns == FEEDBACK_LATENCY_BUDGET_NS
        assert budget.communication_budget_ns == 300
        assert budget.computation_budget_ns == 150
        assert budget.margin_ns == 50

    def test_for_ull_feedback(self):
        budget = LatencyBudget.for_ull_feedback(coherence_time_us=50.0)
        assert budget.total_budget_ns == 50.0  # 50us * 1000 * 0.1%

    def test_for_ull_feedback_custom_coherence(self):
        budget = LatencyBudget.for_ull_feedback(coherence_time_us=100.0)
        assert budget.total_budget_ns == 100.0


class TestEnumerations:
    """Test enum values match HLS constants."""

    def test_reduce_ops(self):
        assert ReduceOp.XOR == 0
        assert ReduceOp.ADD == 1
        assert ReduceOp.MAX == 2
        assert ReduceOp.MIN == 3

    def test_collective_ops(self):
        assert CollectiveOp.BROADCAST == 0
        assert CollectiveOp.BARRIER == 6

    def test_operation_status(self):
        assert OperationStatus.SUCCESS == 0
        assert OperationStatus.UNKNOWN_ERROR == 255

    def test_quantum_msg_types(self):
        assert QuantumMsgType.MEASUREMENT_DATA == 0x10
        assert QuantumMsgType.CONDITIONAL_OP == 0x14


# ============================================================================
# Stats Tests
# ============================================================================

class TestLatencyStats:
    """Test LatencyStats dataclass."""

    def test_from_samples(self):
        samples = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = LatencyStats.from_samples(samples)
        assert stats.count == 5
        assert stats.mean_ns == 300.0
        assert stats.min_ns == 100.0
        assert stats.max_ns == 500.0

    def test_from_empty_samples(self):
        stats = LatencyStats.from_samples([])
        assert stats.count == 0
        assert stats.mean_ns == 0.0

    def test_meets_target_pass(self):
        stats = LatencyStats.from_samples([100.0, 102.0, 98.0, 101.0, 99.0])
        assert stats.meets_target(target_ns=200, jitter_target_ns=10)

    def test_meets_target_fail_mean(self):
        stats = LatencyStats.from_samples([500.0, 600.0, 700.0])
        assert not stats.meets_target(target_ns=200, jitter_target_ns=100)

    def test_str_representation(self):
        stats = LatencyStats.from_samples([100.0])
        s = str(stats)
        assert "LatencyStats" in s
        assert "mean=" in s


class TestLatencyMonitor:
    """Test LatencyMonitor functionality."""

    def test_record_and_stats(self):
        monitor = LatencyMonitor()
        for i in range(10):
            monitor.record(CollectiveOp.BROADCAST, 200 + i, num_ranks=4)

        stats = monitor.get_stats()
        assert CollectiveOp.BROADCAST in stats
        assert stats[CollectiveOp.BROADCAST].count == 10

    def test_violations(self):
        monitor = LatencyMonitor()
        # Broadcast target is 300ns
        monitor.record(CollectiveOp.BROADCAST, 500, num_ranks=4)
        monitor.record(CollectiveOp.BROADCAST, 100, num_ranks=4)

        violations = monitor.get_violations()
        assert violations[CollectiveOp.BROADCAST] == 1

    def test_violation_rate(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.BROADCAST, 500, num_ranks=4)
        monitor.record(CollectiveOp.BROADCAST, 100, num_ranks=4)

        rate = monitor.get_violation_rate(CollectiveOp.BROADCAST)
        assert rate == 0.5

    def test_histogram(self):
        monitor = LatencyMonitor()
        for _ in range(100):
            monitor.record(CollectiveOp.BROADCAST, 200 + np.random.normal(0, 5), num_ranks=4)

        counts, edges = monitor.get_histogram(CollectiveOp.BROADCAST)
        assert len(counts) > 0
        assert len(edges) == len(counts) + 1

    def test_empty_histogram(self):
        monitor = LatencyMonitor()
        counts, edges = monitor.get_histogram(CollectiveOp.BROADCAST)
        assert len(counts) == 0

    def test_clear(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.BROADCAST, 200, num_ranks=4)
        monitor.clear()
        stats = monitor.get_stats()
        assert len(stats) == 0

    def test_export_history(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.REDUCE, 350, num_ranks=8, root_rank=0)
        history = monitor.export_history()
        assert len(history) == 1
        assert history[0]['operation'] == 'REDUCE'
        assert history[0]['latency_ns'] == 350

    def test_alert_callback(self):
        monitor = LatencyMonitor()
        alerts = []
        monitor.add_alert_callback(lambda op, lat, target: alerts.append((op, lat)))

        monitor.record(CollectiveOp.BROADCAST, 500, num_ranks=4)
        assert len(alerts) == 1

    def test_summary(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.BROADCAST, 200, num_ranks=4)
        summary = monitor.summary()
        assert "BROADCAST" in summary

    def test_get_stats_single_op(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.BROADCAST, 200, num_ranks=4)
        monitor.record(CollectiveOp.REDUCE, 350, num_ranks=4)
        stats = monitor.get_stats(CollectiveOp.BROADCAST)
        assert CollectiveOp.BROADCAST in stats
        assert CollectiveOp.REDUCE not in stats


class TestLatencyProfiler:
    """Test LatencyProfiler context manager."""

    def test_profiler_context(self):
        monitor = LatencyMonitor()
        with LatencyProfiler(monitor, CollectiveOp.BROADCAST, num_ranks=4):
            time.sleep(0.001)

        stats = monitor.get_stats()
        assert CollectiveOp.BROADCAST in stats
        assert stats[CollectiveOp.BROADCAST].count == 1
        assert stats[CollectiveOp.BROADCAST].mean_ns > 0


# ============================================================================
# Profiler Tests
# ============================================================================

class TestCriticalPathProfiler:
    """Test CriticalPathProfiler."""

    def test_start_end_operation(self):
        profiler = CriticalPathProfiler()
        op_id = profiler.start_operation('broadcast')
        time.sleep(0.001)
        duration = profiler.end_operation(op_id)
        assert duration is not None
        assert duration > 0

    def test_end_unknown_operation(self):
        profiler = CriticalPathProfiler()
        assert profiler.end_operation('nonexistent') is None

    def test_record_phase(self):
        profiler = CriticalPathProfiler()
        profiler.record_phase('broadcast', 'tree_down', 180.0)
        profiler.record_phase('broadcast', 'serialize', 50.0)

        breakdown = profiler.get_breakdown('broadcast')
        assert 'tree_down' in breakdown.phases
        assert breakdown.phases['tree_down'] == 180.0

    def test_get_breakdown_empty(self):
        profiler = CriticalPathProfiler()
        breakdown = profiler.get_breakdown('nonexistent')
        assert breakdown.total_ns == 0

    def test_get_critical_path(self):
        profiler = CriticalPathProfiler()
        profiler.record_phase('broadcast', 'tree_down', 180.0)
        profiler.record_phase('broadcast', 'serialize', 50.0)

        path = profiler.get_critical_path('broadcast')
        assert len(path) == 2
        assert path[0][0] == 'tree_down'  # Highest first

    def test_clear(self):
        profiler = CriticalPathProfiler()
        profiler.record_phase('broadcast', 'tree_down', 180.0)
        profiler.clear()
        breakdown = profiler.get_breakdown('broadcast')
        assert breakdown.total_ns == 0

    def test_ull_phases_defined(self):
        profiler = CriticalPathProfiler()
        assert 'ull_feedback' in profiler._operation_phases
        assert 'readout' in profiler._operation_phases['ull_feedback']
        assert 'trigger' in profiler._operation_phases['ull_feedback']

    def test_operation_phases_completeness(self):
        profiler = CriticalPathProfiler()
        expected_ops = ['broadcast', 'reduce', 'allreduce', 'barrier',
                        'scatter', 'gather', 'feedback',
                        'ull_feedback', 'ull_broadcast', 'ull_reduce']
        for op in expected_ops:
            assert op in profiler._operation_phases


class TestLatencyBreakdown:
    """Test LatencyBreakdown dataclass."""

    def test_overhead(self):
        bd = LatencyBreakdown(total_ns=300.0, phases={'a': 100.0, 'b': 150.0})
        assert bd.overhead_ns == 50.0

    def test_percentage(self):
        bd = LatencyBreakdown(total_ns=200.0, phases={'a': 100.0})
        assert bd.percentage('a') == 50.0
        assert bd.percentage('nonexistent') == 0.0

    def test_percentage_zero_total(self):
        bd = LatencyBreakdown(total_ns=0)
        assert bd.percentage('anything') == 0.0

    def test_to_dict(self):
        bd = LatencyBreakdown(total_ns=100.0, phases={'x': 60.0})
        d = bd.to_dict()
        assert d['total_ns'] == 100.0
        assert d['phases']['x'] == 60.0
        assert d['overhead_ns'] == 40.0


class TestBottleneckAnalyzer:
    """Test BottleneckAnalyzer."""

    def test_no_data_no_bottlenecks(self):
        profiler = CriticalPathProfiler()
        analyzer = BottleneckAnalyzer(profiler)
        assert len(analyzer.analyze()) == 0

    def test_network_bottleneck_detection(self):
        profiler = CriticalPathProfiler()
        # Record total
        op_id = profiler.start_operation('broadcast')
        profiler._samples.append(ProfileSample(
            timestamp_ns=0, operation='broadcast', phase='total', duration_ns=300.0))
        profiler.record_phase('broadcast', 'tree_down', 250.0)
        profiler.record_phase('broadcast', 'serialize', 30.0)

        analyzer = BottleneckAnalyzer(profiler)
        bottlenecks = analyzer.analyze()
        network_bns = [b for b in bottlenecks if b.type == BottleneckType.NETWORK_LATENCY]
        assert len(network_bns) > 0

    def test_get_summary(self):
        profiler = CriticalPathProfiler()
        analyzer = BottleneckAnalyzer(profiler)
        summary = analyzer.get_summary()
        assert 'total_bottlenecks' in summary


class TestOptimizationAdvisor:
    """Test OptimizationAdvisor."""

    def test_no_recommendations_when_clean(self):
        profiler = CriticalPathProfiler()
        analyzer = BottleneckAnalyzer(profiler)
        advisor = OptimizationAdvisor(analyzer)
        recs = advisor.get_recommendations()
        assert len(recs) == 0

    def test_top_recommendations(self):
        profiler = CriticalPathProfiler()
        analyzer = BottleneckAnalyzer(profiler)
        advisor = OptimizationAdvisor(analyzer)
        top = advisor.get_top_recommendations(n=3)
        assert len(top) <= 3


class TestPerformanceRegressor:
    """Test PerformanceRegressor."""

    def test_no_regressions_no_baseline(self):
        regressor = PerformanceRegressor()
        regressor.update_current('broadcast', LatencyStats.from_samples([200, 210, 190]))
        assert len(regressor.check_regressions()) == 0

    def test_detect_regression(self):
        regressor = PerformanceRegressor()
        regressor._baseline['broadcast'] = LatencyStats.from_samples([200, 210, 190])
        regressor.update_current('broadcast', LatencyStats.from_samples([400, 410, 390]))
        regressions = regressor.check_regressions()
        assert len(regressions) > 0
        assert regressions[0]['operation'] == 'broadcast'

    def test_save_load_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"

            # Save baseline first (file doesn't exist yet)
            regressor = PerformanceRegressor(baseline_path=path)
            regressor.update_current('broadcast', LatencyStats.from_samples([200, 210, 190]))
            regressor.save_baseline()

            # Load in new instance (now file exists with valid JSON)
            regressor2 = PerformanceRegressor(baseline_path=path)
            assert 'broadcast' in regressor2._baseline

    def test_get_comparison(self):
        regressor = PerformanceRegressor()
        regressor._baseline['broadcast'] = LatencyStats.from_samples([200])
        regressor.update_current('broadcast', LatencyStats.from_samples([250]))
        comparison = regressor.get_comparison()
        assert 'broadcast' in comparison
        assert 'changes' in comparison['broadcast']

    def test_update_from_monitor(self):
        monitor = LatencyMonitor()
        monitor.record(CollectiveOp.BROADCAST, 200, num_ranks=4)
        regressor = PerformanceRegressor()
        regressor.update_from_monitor(monitor)
        assert 'BROADCAST' in regressor._current


class TestLatencyVisualizer:
    """Test LatencyVisualizer."""

    def test_breakdown_bar(self):
        bd = LatencyBreakdown(total_ns=300.0, phases={'net': 200.0, 'serial': 80.0})
        result = LatencyVisualizer.breakdown_bar(bd)
        assert "300.0ns" in result
        assert "net" in result

    def test_breakdown_bar_no_data(self):
        bd = LatencyBreakdown(total_ns=0)
        assert "No data" in LatencyVisualizer.breakdown_bar(bd)

    def test_histogram(self):
        samples = [100 + np.random.normal(0, 10) for _ in range(100)]
        result = LatencyVisualizer.histogram(samples)
        assert "n=100" in result

    def test_histogram_no_data(self):
        assert "No data" in LatencyVisualizer.histogram([])

    def test_comparison_table(self):
        comparison = {
            'broadcast': {
                'baseline': {'mean_ns': 200},
                'current': {'mean_ns': 250},
                'changes': {'mean_percent': 25.0},
            }
        }
        result = LatencyVisualizer.comparison_table(comparison)
        assert "broadcast" in result


class TestProfilingSession:
    """Test ProfilingSession."""

    def test_profile_operation_context(self):
        session = ProfilingSession()
        with session.profile_operation('broadcast'):
            time.sleep(0.001)

        breakdown = session.profiler.get_breakdown('broadcast')
        assert breakdown.total_ns > 0

    def test_analyze(self):
        session = ProfilingSession()
        result = session.analyze()
        assert 'session_duration_ns' in result
        assert 'bottlenecks' in result
        assert 'recommendations' in result

    def test_generate_report(self):
        session = ProfilingSession()
        report = session.generate_report()
        assert "ACCL-Q PERFORMANCE PROFILING REPORT" in report


# ============================================================================
# Deployment Tests
# ============================================================================

class TestBoardConfig:
    """Test BoardConfig."""

    def test_to_dict_roundtrip(self):
        board = BoardConfig(
            rank=0, hostname="rfsoc-0",
            ip_address="192.168.1.100",
            mac_address="00:0a:35:00:00:00",
            board_type=BoardType.ZCU216,
        )
        d = board.to_dict()
        assert d['rank'] == 0
        assert d['board_type'] == 'zcu216'

        restored = BoardConfig.from_dict(d)
        assert restored.rank == 0
        assert restored.board_type == BoardType.ZCU216
        assert restored.hostname == "rfsoc-0"

    def test_default_values(self):
        board = BoardConfig(
            rank=1, hostname="test",
            ip_address="10.0.0.1",
            mac_address="aa:bb:cc:dd:ee:ff",
            board_type=BoardType.RFSoC4x2,
        )
        assert board.aurora_lanes == 4
        assert board.dac_channels == 8
        assert board.clock_source == "internal"
        assert board.is_online is False


class TestDeploymentConfig:
    """Test DeploymentConfig."""

    def test_validate_valid(self):
        config = create_default_deployment(num_boards=4)
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_too_few_boards(self):
        config = DeploymentConfig(name="test", num_boards=1)
        errors = config.validate()
        assert any("Minimum 2" in e for e in errors)

    def test_validate_too_many_boards(self):
        config = DeploymentConfig(name="test", num_boards=MAX_RANKS + 1)
        errors = config.validate()
        assert any("Maximum" in e for e in errors)

    def test_validate_master_rank_out_of_range(self):
        config = DeploymentConfig(name="test", num_boards=4, master_rank=5)
        errors = config.validate()
        assert any("Master rank" in e for e in errors)

    def test_save_load_roundtrip(self):
        config = create_default_deployment(num_boards=4, name="test-roundtrip")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            config.save(path)
            loaded = DeploymentConfig.load(path)
            assert loaded.name == "test-roundtrip"
            assert loaded.num_boards == 4
            assert len(loaded.boards) == 4
            assert len(loaded.links) > 0
        finally:
            path.unlink(missing_ok=True)

    def test_min_links_topologies(self):
        for topo, expected_min in [
            (NetworkTopology.STAR, 3),     # n-1 = 3 for 4 boards
            (NetworkTopology.RING, 4),     # n = 4
            (NetworkTopology.TREE, 3),     # n-1 = 3
            (NetworkTopology.FULL_MESH, 6),  # 4*3/2 = 6
        ]:
            config = DeploymentConfig(name="test", num_boards=4, topology=topo)
            assert config._min_links_for_topology() == expected_min


class TestTopologyBuilder:
    """Test TopologyBuilder."""

    def _make_boards(self, n):
        return [
            BoardConfig(
                rank=i, hostname=f"board-{i}",
                ip_address=f"10.0.0.{i}",
                mac_address=f"00:00:00:00:00:{i:02x}",
                board_type=BoardType.ZCU216,
            )
            for i in range(n)
        ]

    def test_build_star(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_star(boards, center_rank=0)
        # Each non-center board has bidirectional link = 2 * 3 = 6
        assert len(links) == 6

    def test_build_ring(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_ring(boards)
        assert len(links) == 4

    def test_build_tree(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_tree(boards, root_rank=0, fanout=4)
        # 3 child nodes * 2 (bidirectional) = 6
        assert len(links) == 6

    def test_build_full_mesh(self):
        boards = self._make_boards(4)
        links = TopologyBuilder.build_full_mesh(boards)
        # 4C2 * 2 = 12 bidirectional links
        assert len(links) == 12


class TestDeploymentManager:
    """Test DeploymentManager basic functionality (no real network)."""

    def test_initial_state(self):
        config = create_default_deployment(num_boards=4)
        manager = DeploymentManager(config)
        assert manager.state == DeploymentState.UNINITIALIZED

    def test_get_status(self):
        config = create_default_deployment(num_boards=4)
        manager = DeploymentManager(config)
        status = manager.get_status()
        assert status['state'] == 'uninitialized'
        assert status['num_boards'] == 4
        assert status['online_boards'] == 0

    def test_state_callbacks(self):
        config = create_default_deployment(num_boards=4)
        manager = DeploymentManager(config)
        states = []
        manager.add_state_callback(lambda s: states.append(s))
        manager._set_state(DeploymentState.CONFIGURING)
        assert states == [DeploymentState.CONFIGURING]

    def test_error_callbacks(self):
        config = create_default_deployment(num_boards=4)
        manager = DeploymentManager(config)
        errors = []
        manager.add_error_callback(lambda msg: errors.append(msg))
        manager._report_error("test error")
        assert "test error" in errors[0]

    def test_shutdown(self):
        config = create_default_deployment(num_boards=4)
        manager = DeploymentManager(config)
        manager.shutdown()
        assert manager.state == DeploymentState.SHUTDOWN


class TestCreateDefaultDeployment:
    """Test create_default_deployment helper."""

    def test_4_boards(self):
        config = create_default_deployment(num_boards=4)
        assert len(config.boards) == 4
        assert config.topology == NetworkTopology.TREE
        assert len(config.links) > 0

    def test_8_boards(self):
        config = create_default_deployment(num_boards=8, name="big-test")
        assert len(config.boards) == 8
        assert config.name == "big-test"

    def test_boards_have_correct_ips(self):
        config = create_default_deployment(num_boards=4)
        for rank in range(4):
            assert config.boards[rank].ip_address == f"192.168.1.{100 + rank}"


class TestBoardType:
    """Test BoardType enum."""

    def test_all_types(self):
        types = list(BoardType)
        assert len(types) == 6  # ZCU111, ZCU216, RFSoC2x2, RFSoC4x2, HTGZRF16, CUSTOM


class TestNetworkTopology:
    """Test NetworkTopology enum."""

    def test_all_topologies(self):
        topos = list(NetworkTopology)
        assert len(topos) == 5


class TestDeploymentState:
    """Test DeploymentState enum."""

    def test_state_values(self):
        assert DeploymentState.UNINITIALIZED.value == "uninitialized"
        assert DeploymentState.READY.value == "ready"
        assert DeploymentState.ERROR.value == "error"
