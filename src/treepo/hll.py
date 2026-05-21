from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from treepo.common import ScheduleName, VALID_SCHEDULES


def _splitmix64(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class HLLConfig:
    precision: int
    hash_bits: int = 64


def _hll_alpha(m: int) -> float:
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / float(m))


class HyperLogLogSketch:
    def __init__(self, config: HLLConfig):
        if not (4 <= int(config.precision) <= int(config.hash_bits) - 2):
            raise ValueError("precision must be in [4, hash_bits-2]")
        self.config = config
        self.m = 1 << int(config.precision)
        self.registers = np.zeros(self.m, dtype=np.uint8)
        self._remaining_bits = int(config.hash_bits) - int(config.precision)
        self._remaining_mask = (1 << self._remaining_bits) - 1

    @property
    def register_bits(self) -> int:
        return int(math.ceil(math.log2(self._remaining_bits + 1)))

    @property
    def memory_bits(self) -> int:
        return int(self.m * self.register_bits)

    @property
    def memory_bytes(self) -> float:
        return float(self.memory_bits) / 8.0

    def copy(self) -> "HyperLogLogSketch":
        out = HyperLogLogSketch(self.config)
        out.registers[:] = self.registers
        return out

    def add(self, token_id: int) -> "HyperLogLogSketch":
        h = _splitmix64(int(token_id))
        p = int(self.config.precision)
        idx = h >> (self.config.hash_bits - p)
        w = h & self._remaining_mask
        if w == 0:
            rho = self._remaining_bits + 1
        else:
            rho = self._remaining_bits - int(w.bit_length()) + 1
        if rho > int(self.registers[idx]):
            self.registers[idx] = min(rho, 255)
        return self

    def update(self, token_ids: Iterable[int]) -> "HyperLogLogSketch":
        for tok in token_ids:
            self.add(int(tok))
        return self

    def merge(self, other: "HyperLogLogSketch") -> "HyperLogLogSketch":
        if self.config != other.config:
            raise ValueError("cannot merge HLL sketches with different configs")
        np.maximum(self.registers, other.registers, out=self.registers)
        return self

    def estimate(self) -> float:
        m = float(self.m)
        z = np.power(2.0, -self.registers.astype(np.float64)).sum()
        raw = _hll_alpha(self.m) * (m * m) / max(z, 1e-12)
        n_zeros = int((self.registers == 0).sum())
        if raw <= 2.5 * m and n_zeros > 0:
            return m * math.log(m / float(n_zeros))

        hash_space = float(2.0 ** int(self.config.hash_bits))
        if raw > hash_space / 30.0:
            clipped = min(raw / hash_space, 1.0 - 1e-12)
            return -hash_space * math.log1p(-clipped)
        return raw

    @staticmethod
    def from_tokens(config: HLLConfig, token_ids: Sequence[int]) -> "HyperLogLogSketch":
        return HyperLogLogSketch(config).update(token_ids)

    @staticmethod
    def from_registers(config: HLLConfig, registers: np.ndarray) -> "HyperLogLogSketch":
        sk = HyperLogLogSketch(config)
        regs = np.asarray(registers, dtype=np.uint8)
        if regs.shape != sk.registers.shape:
            raise ValueError(
                f"register shape mismatch: expected {sk.registers.shape}, got {regs.shape}"
            )
        sk.registers[:] = regs
        return sk


def hll_relative_standard_error(precision: int) -> float:
    return 1.04 / math.sqrt(float(1 << int(precision)))


def match_hll_precision_for_bits(
    target_bits: int,
    *,
    hash_bits: int = 64,
    p_min: int = 4,
    p_max: int = 16,
) -> int:
    target = int(max(1, target_bits))
    best_p = int(p_min)
    best_gap = float("inf")
    for p in range(int(p_min), int(p_max) + 1):
        gap = abs(int(HyperLogLogSketch(HLLConfig(precision=p, hash_bits=hash_bits)).memory_bits) - target)
        if gap < best_gap:
            best_gap = float(gap)
            best_p = int(p)
    return best_p


def reduce_hll_sketches(
    sketches: Sequence[HyperLogLogSketch],
    *,
    schedule: ScheduleName = "balanced",
) -> HyperLogLogSketch:
    if len(sketches) == 0:
        raise ValueError("sketches must be non-empty")
    if any(sk.config != sketches[0].config for sk in sketches):
        raise ValueError("all HLL sketches must share the same config")

    cur = [sk.copy() for sk in sketches]
    sched = str(schedule)
    if sched == "balanced":
        while len(cur) > 1:
            nxt: list[HyperLogLogSketch] = []
            i = 0
            while i < len(cur):
                if i + 1 >= len(cur):
                    nxt.append(cur[i])
                    i += 1
                    continue
                nxt.append(cur[i].copy().merge(cur[i + 1]))
                i += 2
            cur = nxt
        return cur[0]

    if sched in ("left_to_right", "right_to_left"):
        order = cur if sched == "left_to_right" else list(reversed(cur))
        acc = order[0].copy()
        for sk in order[1:]:
            acc.merge(sk)
        return acc

    raise ValueError(f"unsupported schedule: {schedule!r}; expected one of {VALID_SCHEDULES}")


__all__ = [
    "HLLConfig",
    "HyperLogLogSketch",
    "VALID_SCHEDULES",
    "hll_relative_standard_error",
    "match_hll_precision_for_bits",
    "reduce_hll_sketches",
]
