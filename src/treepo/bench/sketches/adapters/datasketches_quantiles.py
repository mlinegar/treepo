"""Quantile sketch adapters backed by Apache DataSketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from treepo.bench.sketches.adapters._datasketches import _ds, _require_datasketches


class _QuantilesAdapterBase:
    """Shared deserialize-copy merge surface for the quantile adapters."""

    is_commutative = True
    is_associative = True
    is_idempotent = False
    is_byte_deterministic = False

    def _new_sketch(self) -> Any:
        raise NotImplementedError

    def _sketch_type(self) -> Any:
        raise NotImplementedError

    def empty(self) -> Any:
        _require_datasketches()
        return self._new_sketch()

    def encode(self, items: Iterable[float]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(float(item))
        return sk

    def merge(self, a: Any, b: Any) -> Any:
        _require_datasketches()
        out = self._sketch_type().deserialize(bytes(a.serialize()))
        out.merge(b)
        return out

    def query(self, s: Any, q: float) -> float:
        return float(s.get_quantile(float(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def _count(self, s: Any) -> int:
        return int(s.n)

    def state_equal(self, a: Any, b: Any) -> bool:
        return self._count(a) == self._count(b) and abs(
            float(a.get_quantile(0.5)) - float(b.get_quantile(0.5))
        ) < 1e-6

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class KLLFloatsDatasketchesAdapter(_QuantilesAdapterBase):
    """Apache DataSketches KLL float quantile adapter."""

    k: int = 200

    name: str = "kll_floats_datasketches"

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "kll_floats", "k": int(self.k)}

    def _new_sketch(self) -> Any:
        return _ds.kll_floats_sketch(int(self.k))

    def _sketch_type(self) -> Any:
        return _ds.kll_floats_sketch


@dataclass(frozen=True)
class QuantilesFloatsDatasketchesAdapter(_QuantilesAdapterBase):
    """Apache DataSketches classic quantiles float adapter."""

    k: int = 128

    name: str = "quantiles_floats_datasketches"

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "quantiles_floats", "k": int(self.k)}

    def _new_sketch(self) -> Any:
        return _ds.quantiles_floats_sketch(int(self.k))

    def _sketch_type(self) -> Any:
        return _ds.quantiles_floats_sketch


@dataclass(frozen=True)
class REQFloatsDatasketchesAdapter(_QuantilesAdapterBase):
    """Apache DataSketches REQ float quantile adapter."""

    k: int = 12
    high_rank_accuracy: bool = True

    name: str = "req_floats_datasketches"

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "req_floats",
            "k": int(self.k),
            "high_rank_accuracy": bool(self.high_rank_accuracy),
        }

    def _new_sketch(self) -> Any:
        return _ds.req_floats_sketch(int(self.k), bool(self.high_rank_accuracy))

    def _sketch_type(self) -> Any:
        return _ds.req_floats_sketch


@dataclass(frozen=True)
class TDigestDoubleDatasketchesAdapter(_QuantilesAdapterBase):
    """Apache DataSketches t-digest adapter."""

    k: int = 200

    name: str = "tdigest_double_datasketches"

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "tdigest_double", "k": int(self.k)}

    def _new_sketch(self) -> Any:
        return _ds.tdigest_double(int(self.k))

    def _sketch_type(self) -> Any:
        return _ds.tdigest_double

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes())

    def _count(self, s: Any) -> int:
        return int(s.get_total_weight())
