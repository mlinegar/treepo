"""`CardinalitySketch` adapter wrapping `treepo.hll.HyperLogLogSketch`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from treepo.hll import HLLConfig, HyperLogLogSketch


@dataclass(frozen=True)
class HLLNativeAdapter:
    """Wraps the repo's native HLL implementation.

    State is a raw `HyperLogLogSketch` (register numpy array + config). Merge
    is the classical register-wise max; `state_equal` is byte-equality on the
    register array.
    """

    precision: int
    hash_bits: int = 64

    name: str = "hll_native"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = True
    is_byte_deterministic: bool = True

    @property
    def config(self) -> dict:
        return {
            "backend": "native",
            "precision": int(self.precision),
            "hash_bits": int(self.hash_bits),
        }

    def _cfg(self) -> HLLConfig:
        return HLLConfig(precision=int(self.precision), hash_bits=int(self.hash_bits))

    def empty(self) -> HyperLogLogSketch:
        return HyperLogLogSketch(self._cfg())

    def update(self, s: HyperLogLogSketch, item: int) -> HyperLogLogSketch:
        s.add(int(item))
        return s

    def encode(self, items: Iterable[int]) -> HyperLogLogSketch:
        return HyperLogLogSketch.from_tokens(self._cfg(), list(items))

    def merge(self, a: HyperLogLogSketch, b: HyperLogLogSketch) -> HyperLogLogSketch:
        out = a.copy()
        out.merge(b)
        return out

    def query(self, s: HyperLogLogSketch, q: None = None) -> float:
        return float(s.estimate())

    def serialize(self, s: HyperLogLogSketch) -> bytes:
        return bytes(s.registers.tobytes())

    def serialized_size_bytes(self, s: HyperLogLogSketch) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: HyperLogLogSketch, b: HyperLogLogSketch) -> bool:
        if a.config != b.config:
            return False
        return bool(np.array_equal(a.registers, b.registers))

    def memory_bytes(self, s: HyperLogLogSketch) -> float:
        return float(s.memory_bytes)
