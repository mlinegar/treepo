"""Standard mergeable-sketch baselines for comparison with the learned sketch.

We include the two canonical baselines for this setting:

* `HyperLogLogSketch` — the field-standard mergeable sketch for estimating
  cardinality of a set. Here we estimate the number of *distinct bigrams*.
  Mergeable via per-register max.
* Analytic `BigramSketch` — the exact mergeable sketch from `sketch_data.py`.
  This is the zero-error gold standard on the bigram-count task.

Both are evaluated on held-out sequences and reported alongside the learned
MLP merge operator's results in the training summary, so the sanity check
shows how the learned sketch compares to "what everyone already uses".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence


def _hash_bigram(a: int, b: int) -> int:
    payload = f"{int(a)}|{int(b)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


@dataclass
class HyperLogLogSketch:
    """Tiny mergeable HyperLogLog (base implementation, no small-range correction).

    Register count `m = 2^p`. We use p=6 -> 64 registers by default, which gives
    ~13% standard error — fine for a sanity-check baseline at this scale.
    """

    registers: list[int]
    p: int

    @classmethod
    def empty(cls, *, p: int = 6) -> "HyperLogLogSketch":
        return cls(registers=[0] * (1 << int(p)), p=int(p))

    @classmethod
    def from_bigrams(
        cls, bigrams: Sequence[tuple[int, int]], *, p: int = 6
    ) -> "HyperLogLogSketch":
        sketch = cls.empty(p=p)
        for a, b in bigrams:
            sketch.add(a, b)
        return sketch

    def add(self, a: int, b: int) -> None:
        h = _hash_bigram(a, b)
        # Top p bits pick the register, remaining 64-p bits encode the leading-zero run.
        idx = h >> (64 - self.p)
        remainder = (h << self.p) & ((1 << 64) - 1)
        if remainder == 0:
            rank = 64 - self.p + 1
        else:
            rank = 0
            bit = 1 << 63
            while remainder & bit == 0 and rank < 64 - self.p:
                rank += 1
                bit >>= 1
            rank += 1
        if rank > self.registers[idx]:
            self.registers[idx] = rank

    def merge(self, other: "HyperLogLogSketch") -> "HyperLogLogSketch":
        if self.p != other.p:
            raise ValueError("cannot merge HyperLogLogSketch with differing precision")
        merged = [max(x, y) for x, y in zip(self.registers, other.registers)]
        return HyperLogLogSketch(registers=merged, p=self.p)

    def estimate(self) -> float:
        m = len(self.registers)
        # Alpha-m constant per Flajolet et al.
        if m == 16:
            alpha = 0.673
        elif m == 32:
            alpha = 0.697
        elif m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / m)
        z = sum(2.0 ** (-r) for r in self.registers)
        raw = alpha * m * m / max(z, 1e-12)
        # Small-range linear-counting correction when many registers are zero.
        zero_count = sum(1 for r in self.registers if r == 0)
        if raw <= 2.5 * m and zero_count > 0:
            import math

            return float(m * math.log(m / zero_count))
        return float(raw)


def bigrams_of(tokens: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in zip(tokens, tokens[1:])]


def true_distinct_bigrams(tokens: Sequence[int]) -> int:
    return len({(int(a), int(b)) for a, b in zip(tokens, tokens[1:])})
