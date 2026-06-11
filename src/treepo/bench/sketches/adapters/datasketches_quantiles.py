"""Quantile sketch adapters backed by Apache DataSketches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import datasketches as _ds
except ImportError:  # pragma: no cover
    _ds = None  # type: ignore[assignment]


def _require_datasketches() -> None:
    if _ds is None:
        raise ImportError(
            "datasketches is required for official sketch benchmarks. "
            "Install with: uv sync --extra sketches"
        )


@dataclass(frozen=True)
class KLLFloatsDatasketchesAdapter:
    """Apache DataSketches KLL float quantile adapter."""

    k: int = 200

    name: str = "kll_floats_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "kll_floats", "k": int(self.k)}

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.kll_floats_sketch(int(self.k))

    def update(self, s: Any, item: float) -> Any:
        s.update(float(item))
        return s

    def encode(self, items: Iterable[float]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(float(item))
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.kll_floats_sketch.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: float) -> float:
        return float(s.get_quantile(float(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        return int(a.n) == int(b.n) and abs(float(a.get_quantile(0.5)) - float(b.get_quantile(0.5))) < 1e-6

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class QuantilesFloatsDatasketchesAdapter:
    """Apache DataSketches classic quantiles float adapter."""

    k: int = 128

    name: str = "quantiles_floats_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "quantiles_floats", "k": int(self.k)}

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.quantiles_floats_sketch(int(self.k))

    def update(self, s: Any, item: float) -> Any:
        s.update(float(item))
        return s

    def encode(self, items: Iterable[float]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(float(item))
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.quantiles_floats_sketch.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: float) -> float:
        return float(s.get_quantile(float(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        return int(a.n) == int(b.n) and abs(float(a.get_quantile(0.5)) - float(b.get_quantile(0.5))) < 1e-6

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class REQFloatsDatasketchesAdapter:
    """Apache DataSketches REQ float quantile adapter."""

    k: int = 12
    high_rank_accuracy: bool = True

    name: str = "req_floats_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {
            "backend": "datasketches",
            "family": "req_floats",
            "k": int(self.k),
            "high_rank_accuracy": bool(self.high_rank_accuracy),
        }

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.req_floats_sketch(int(self.k), bool(self.high_rank_accuracy))

    def update(self, s: Any, item: float) -> Any:
        s.update(float(item))
        return s

    def encode(self, items: Iterable[float]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(float(item))
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.req_floats_sketch.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: float) -> float:
        return float(s.get_quantile(float(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(len(self.serialize(s)))

    def state_equal(self, a: Any, b: Any) -> bool:
        return int(a.n) == int(b.n) and abs(float(a.get_quantile(0.5)) - float(b.get_quantile(0.5))) < 1e-6

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)


@dataclass(frozen=True)
class TDigestDoubleDatasketchesAdapter:
    """Apache DataSketches t-digest adapter."""

    k: int = 200

    name: str = "tdigest_double_datasketches"
    is_commutative: bool = True
    is_associative: bool = True
    is_idempotent: bool = False
    is_byte_deterministic: bool = False

    @property
    def config(self) -> dict:
        return {"backend": "datasketches", "family": "tdigest_double", "k": int(self.k)}

    def empty(self) -> Any:
        _require_datasketches()
        return _ds.tdigest_double(int(self.k))

    def update(self, s: Any, item: float) -> Any:
        s.update(float(item))
        return s

    def encode(self, items: Iterable[float]) -> Any:
        sk = self.empty()
        for item in items:
            sk.update(float(item))
        return sk

    def _copy(self, s: Any) -> Any:
        _require_datasketches()
        return _ds.tdigest_double.deserialize(bytes(s.serialize()))

    def merge(self, a: Any, b: Any) -> Any:
        out = self._copy(a)
        out.merge(b)
        return out

    def query(self, s: Any, q: float) -> float:
        return float(s.get_quantile(float(q)))

    def serialize(self, s: Any) -> bytes:
        return bytes(s.serialize())

    def serialized_size_bytes(self, s: Any) -> float:
        return float(s.get_serialized_size_bytes())

    def state_equal(self, a: Any, b: Any) -> bool:
        return int(a.get_total_weight()) == int(b.get_total_weight()) and abs(
            float(a.get_quantile(0.5)) - float(b.get_quantile(0.5))
        ) < 1e-6

    def memory_bytes(self, s: Any) -> float:
        return self.serialized_size_bytes(s)
