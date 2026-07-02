"""`SketchAdapter` Protocol — uniform surface over classical mergeable sketches.

Concrete adapters (HLL/CPC/Theta/Count-Min/FrequentItems/KLL/REQ/t-digest/
Tuple/VarOpt from Apache DataSketches) implement this Protocol. The tree
reducer and benchmark runners operate against the Protocol; concrete adapters
gate the `datasketches` import.

The `state` type parameter is the sketch's internal representation (register
array, bitmap, centroid list, etc.). `Item` is the type fed per leaf (int
token ids for HLL, hashables for Theta, hashables or ``(key, weight)`` pairs
for Count-Min/Frequent Items, floats for KLL/REQ/t-digest, and weighted items
for Tuple/VarOpt).
`Query` and `Result` cover what the sketch answers: cardinality float for HLL;
a frequency estimate for Count-Min/Frequent Items; a quantile for
KLL/REQ/t-digest.
"""

from __future__ import annotations

from typing import Any, Iterable, Protocol, TypeVar, runtime_checkable

StateT = TypeVar("StateT")
ItemT = TypeVar("ItemT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")


@runtime_checkable
class SketchAdapter(Protocol[StateT, ItemT, QueryT, ResultT]):
    """Uniform surface over a classical mergeable sketch.

    Contract:
    - `empty()` is the identity element of `merge`.
    - `merge` is associative; it is often commutative, and idempotence is
      sketch-specific (for example HLL/Theta-style unions, but not Count-Min).
      Adapters declare this via `is_commutative` and `is_idempotent`.
    - `merge(a, b)` must not mutate its inputs. Adapters copy as needed.
    - `serialize(s)` must be deterministic so `state_equal` via bytes is a
      valid check when `is_byte_deterministic=True`.
    """

    name: str
    config: dict[str, Any]
    is_commutative: bool
    is_associative: bool
    is_idempotent: bool
    is_byte_deterministic: bool

    def empty(self) -> StateT: ...

    def encode(self, items: Iterable[ItemT]) -> StateT: ...

    def merge(self, a: StateT, b: StateT) -> StateT: ...

    def query(self, s: StateT, q: QueryT) -> ResultT: ...

    def serialize(self, s: StateT) -> bytes: ...

    def serialized_size_bytes(self, s: StateT) -> float: ...

    def state_equal(self, a: StateT, b: StateT) -> bool: ...

    def memory_bytes(self, s: StateT) -> float: ...


@runtime_checkable
class CardinalitySketch(SketchAdapter[StateT, ItemT, None, float], Protocol):
    """Sub-Protocol for cardinality sketches (HLL, Theta/KMV).

    `query(s, None) -> float` returns the estimated distinct-item count.
    """
