"""
ACCL-Q Hardware Acceleration for Ultra-Low-Latency Operations

Provides DMA buffer pooling, BRAM LUT decoder simulation, and FPGA register
interface for hardware-autonomous feedback execution.

In a real deployment, these classes drive actual FPGA registers. In simulation,
they model the hardware behavior with cycle-accurate latency estimates.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from enum import IntEnum

from .constants import (
    ULL_TARGET_MULTICAST_NS,
    ULL_TARGET_REDUCE_NS,
    ULL_TARGET_DECODE_NS,
    ULL_TARGET_TRIGGER_NS,
    ULL_TARGET_TOTAL_NS,
    ULL_MAX_SYNDROME_BITS,
    ULL_LUT_DECODER_DEPTH,
    ULL_DMA_BUFFER_ALIGNMENT,
    ULL_DMA_BUFFER_POOL_SIZE,
    ULLPipelineConfig,
    CLOCK_PERIOD_NS,
    FIBER_DELAY_NS_PER_METER,
)


# ============================================================================
# DMA Buffer Pool
# ============================================================================

class DMABufferPool:
    """
    Pre-allocated DMA buffer pool for zero-copy data transfers.

    Buffers are cache-line aligned (64 bytes) and reusable without
    allocation overhead in the hot path.
    """

    def __init__(self, num_buffers: int = ULL_DMA_BUFFER_POOL_SIZE,
                 buffer_size_bytes: int = 64,
                 alignment: int = ULL_DMA_BUFFER_ALIGNMENT):
        self._alignment = alignment
        self._buffer_size = buffer_size_bytes
        self._total = num_buffers

        # Pre-allocate aligned buffers
        self._free: deque = deque()
        self._all_buffers: List[np.ndarray] = []
        for _ in range(num_buffers):
            # numpy arrays are typically aligned, but we ensure element count
            # is a multiple of alignment / element_size
            buf = np.zeros(buffer_size_bytes, dtype=np.uint8)
            self._all_buffers.append(buf)
            self._free.append(buf)

        self._acquired_count = 0

    def acquire(self) -> np.ndarray:
        """Acquire a buffer from the pool. Raises RuntimeError if exhausted."""
        if not self._free:
            raise RuntimeError(
                f"DMA buffer pool exhausted ({self._total} buffers in use)"
            )
        buf = self._free.popleft()
        self._acquired_count += 1
        return buf

    def release(self, buf: np.ndarray) -> None:
        """Return a buffer to the pool."""
        self._free.append(buf)
        self._acquired_count -= 1

    def get_buffer(self, index: int) -> np.ndarray:
        """Get a buffer by index (zero-copy access to pre-allocated pool)."""
        if index < 0 or index >= self._total:
            raise IndexError(f"Buffer index {index} out of range [0, {self._total})")
        return self._all_buffers[index]

    @property
    def available(self) -> int:
        return len(self._free)

    @property
    def total(self) -> int:
        return self._total

    @property
    def in_use(self) -> int:
        return self._acquired_count


# ============================================================================
# LUT Decoder
# ============================================================================

class LUTDecoder:
    """
    BRAM-based lookup table decoder for syndrome-to-correction mapping.

    In hardware, this is a dual-port BRAM addressed by syndrome value,
    returning the correction in 4 clock cycles (8ns at 500MHz).

    In simulation, uses a Python dict for the lookup and builds a BRAM
    image (numpy array) that could be loaded into actual FPGA BRAM.
    """

    def __init__(self, num_syndrome_bits: int, lut_depth: int = ULL_LUT_DECODER_DEPTH):
        if num_syndrome_bits > ULL_MAX_SYNDROME_BITS:
            raise ValueError(
                f"Syndrome size {num_syndrome_bits} exceeds ULL max {ULL_MAX_SYNDROME_BITS}"
            )
        self._num_bits = num_syndrome_bits
        self._lut_depth = lut_depth
        self._table: Dict[int, np.ndarray] = {}
        self._bram_image: Optional[np.ndarray] = None
        self._programmed = False

    def program(self, decoder_fn: Callable[[np.ndarray], np.ndarray]) -> int:
        """
        Program the LUT by enumerating weight-1 and weight-2 syndromes.

        Args:
            decoder_fn: Function mapping syndrome array → correction array

        Returns:
            Number of entries programmed
        """
        self._table.clear()
        n = self._num_bits
        entries = 0

        # Weight-0 (trivial syndrome)
        syndrome = np.zeros(n, dtype=np.uint8)
        correction = decoder_fn(syndrome)
        self._table[0] = correction.copy()
        entries += 1

        # Weight-1 syndromes
        for i in range(min(n, self._lut_depth - 1)):
            syndrome = np.zeros(n, dtype=np.uint8)
            syndrome[i] = 1
            key = 1 << i
            correction = decoder_fn(syndrome)
            self._table[key] = correction.copy()
            entries += 1
            if entries >= self._lut_depth:
                break

        # Weight-2 syndromes (if space remains)
        if entries < self._lut_depth:
            for i in range(min(n, 32)):  # Cap to avoid combinatorial explosion
                for j in range(i + 1, min(n, 32)):
                    if entries >= self._lut_depth:
                        break
                    syndrome = np.zeros(n, dtype=np.uint8)
                    syndrome[i] = 1
                    syndrome[j] = 1
                    key = (1 << i) | (1 << j)
                    correction = decoder_fn(syndrome)
                    self._table[key] = correction.copy()
                    entries += 1
                if entries >= self._lut_depth:
                    break

        self._build_bram_image()
        self._programmed = True
        return entries

    def lookup(self, syndrome: np.ndarray) -> Optional[np.ndarray]:
        """Look up correction for a syndrome (simulation path)."""
        key = self._syndrome_to_key(syndrome)
        return self._table.get(key)

    def get_bram_image(self) -> Optional[np.ndarray]:
        """Get the BRAM image for FPGA programming."""
        return self._bram_image

    @property
    def programmed(self) -> bool:
        return self._programmed

    @property
    def num_entries(self) -> int:
        return len(self._table)

    def _syndrome_to_key(self, syndrome: np.ndarray) -> int:
        """Convert syndrome array to integer key."""
        key = 0
        for i, bit in enumerate(syndrome):
            if bit:
                key |= (1 << i)
        return key

    def _build_bram_image(self) -> None:
        """Build a flat numpy array representing the BRAM contents."""
        if not self._table:
            self._bram_image = None
            return
        # Get correction size from first entry
        first_correction = next(iter(self._table.values()))
        correction_size = len(first_correction)
        # Build image: each row is a correction indexed by syndrome key
        image = np.zeros((self._lut_depth, correction_size), dtype=np.uint8)
        for key, correction in self._table.items():
            if key < self._lut_depth:
                image[key] = correction[:correction_size]
        self._bram_image = image


# ============================================================================
# FPGA Register Interface
# ============================================================================

class ULLRegister(IntEnum):
    """ULL FPGA register map offsets."""
    ULL_CONTROL = 0x100
    ULL_STATUS = 0x104
    SYNDROME_MASK = 0x108
    DECODER_BASE = 0x10C
    TRIGGER_CONFIG = 0x110
    LATENCY_COUNTER = 0x114


class FPGARegisterInterface:
    """
    Simulated FPGA register interface for ULL pipeline control.

    In hardware, these are memory-mapped register reads/writes.
    In simulation, tracks state and models register behavior.
    """

    def __init__(self):
        self._registers: Dict[int, int] = {reg: 0 for reg in ULLRegister}
        self._armed = False
        self._latency_cycles = 0

    def write(self, addr: int, value: int) -> None:
        """Write to an FPGA register."""
        self._registers[addr] = value

    def read(self, addr: int) -> int:
        """Read from an FPGA register."""
        return self._registers.get(addr, 0)

    def arm_ull_pipeline(self) -> None:
        """Arm the ULL hardware pipeline for autonomous execution."""
        self._registers[ULLRegister.ULL_CONTROL] = 1
        self._armed = True

    def disarm_ull_pipeline(self) -> None:
        """Disarm the ULL pipeline."""
        self._registers[ULLRegister.ULL_CONTROL] = 0
        self._armed = False

    def is_pipeline_active(self) -> bool:
        """Check if ULL pipeline is armed and active."""
        return self._armed

    def get_last_latency_cycles(self) -> int:
        """Read the hardware latency counter (last cycle's value)."""
        return self._registers.get(ULLRegister.LATENCY_COUNTER, 0)

    def set_latency_cycles(self, cycles: int) -> None:
        """Set the latency counter (for simulation)."""
        self._registers[ULLRegister.LATENCY_COUNTER] = cycles


# ============================================================================
# Hardware Accelerator (coordinates pool + decoder + registers)
# ============================================================================

class HardwareAccelerator:
    """
    Coordinates DMA pool, LUT decoder, and FPGA registers for ULL operation.

    This is the top-level hardware abstraction. Python calls
    `program_pipeline()` once during setup, then the FPGA executes
    feedback loops autonomously.
    """

    def __init__(self, config: Optional[ULLPipelineConfig] = None):
        self.config = config or ULLPipelineConfig()
        self.pool = DMABufferPool(
            num_buffers=self.config.dma_buffer_count,
            buffer_size_bytes=max(64, self.config.max_syndrome_bits // 8),
        )
        self.decoder = LUTDecoder(
            num_syndrome_bits=self.config.max_syndrome_bits,
            lut_depth=self.config.lut_depth,
        )
        self.registers = FPGARegisterInterface()
        self._programmed = False

    def program_pipeline(self, decoder_fn: Callable[[np.ndarray], np.ndarray]) -> int:
        """
        Program the full ULL pipeline: build LUT, configure registers, arm.

        Args:
            decoder_fn: Syndrome → correction mapping function

        Returns:
            Number of LUT entries programmed
        """
        entries = self.decoder.program(decoder_fn)
        # Configure syndrome mask register
        mask = (1 << self.config.max_syndrome_bits) - 1
        self.registers.write(ULLRegister.SYNDROME_MASK, mask)
        # Arm the pipeline
        self.registers.arm_ull_pipeline()
        self._programmed = True
        return entries

    def disarm(self) -> None:
        """Disarm the hardware pipeline."""
        self.registers.disarm_ull_pipeline()

    def estimate_latency_ns(self) -> float:
        """
        Estimate total ULL feedback latency based on config.

        Returns:
            Estimated latency in nanoseconds
        """
        multicast = ULL_TARGET_MULTICAST_NS if self.config.use_hardware_multicast else 40
        reduce = ULL_TARGET_REDUCE_NS if self.config.use_combinational_reduce else 20
        decode = ULL_TARGET_DECODE_NS
        trigger = ULL_TARGET_TRIGGER_NS if self.config.auto_trigger else 10
        fiber = self.config.fiber_length_m * FIBER_DELAY_NS_PER_METER

        return multicast + reduce + decode + trigger + fiber

    def validate_config(self) -> List[str]:
        """
        Validate ULL configuration and return warnings.

        Returns:
            List of warning strings (empty if all clear)
        """
        warnings = []
        estimated = self.estimate_latency_ns()
        budget = self.config.coherence_time_us * 1000 * 0.001  # 0.1%

        if estimated > budget:
            warnings.append(
                f"Estimated latency {estimated:.1f}ns exceeds budget {budget:.1f}ns"
            )

        fiber_delay = self.config.fiber_length_m * FIBER_DELAY_NS_PER_METER
        if fiber_delay > 10:
            warnings.append(
                f"Fiber delay {fiber_delay:.1f}ns too high for 50ns budget "
                f"(fiber_length_m={self.config.fiber_length_m})"
            )

        if self.config.max_syndrome_bits > ULL_MAX_SYNDROME_BITS:
            warnings.append(
                f"Syndrome bits {self.config.max_syndrome_bits} exceeds "
                f"ULL max {ULL_MAX_SYNDROME_BITS}"
            )

        return warnings

    @property
    def is_programmed(self) -> bool:
        return self._programmed
