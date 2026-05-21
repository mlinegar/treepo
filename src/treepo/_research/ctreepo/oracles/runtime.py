"""``OracleFamilyRuntime`` — adapter making any registered oracle a ``FamilyRuntime``.

Synthetic-DGP simulations (Markov / LDA / classical sketches) know their
oracle f* exactly. This runtime lets the alternating ladder treat oracles as
first-class families: ``--family hll_exact`` or ``--family markov_exact``
plug straight into the ladder loop with ``train_f``/``train_g`` as no-ops.

For mixed workflows (the user's "fix f=oracle, learn g" pattern), backend
families like FNO / CTreePO-native should accept ``--f-init oracle:<name>``
via their own ``resolve_init`` and adapt the oracle to their f signature.
This runtime is the case where the oracle IS the family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from treepo._research.ctreepo.alternating import (
    BundleAwareFamilyRuntime,
    FamilyRuntime,
)

from . import OracleSpec, get_oracle


class OracleFamilyRuntime:
    """Family runtime that wraps a registered :class:`OracleSpec`.

    Implements both :class:`FamilyRuntime` and :class:`BundleAwareFamilyRuntime`.
    Training is a no-op: the oracle is the answer. ``score_roots_with_f``
    requires the underlying spec to expose a ``score_tree`` adapter; raises
    ``RuntimeError`` if absent (with an actionable message pointing to the
    spec definition).
    """

    def __init__(self, oracle_name: str) -> None:
        self._spec: OracleSpec = get_oracle(oracle_name)
        self._oracle_name = str(oracle_name)

    # ----------------------------- identity ----------------------------- #

    @property
    def name(self) -> str:
        return f"oracle:{self._oracle_name}"

    @property
    def spec(self) -> OracleSpec:
        return self._spec

    # ------------------------ FamilyRuntime API ------------------------- #

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        # Oracle f* is fixed by construction; training is a no-op.
        return self._oracle_handle()

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        # Oracle families that have a g_callable (e.g. hll_max_merge) keep it
        # fixed. Families whose g is raw_concat keep the sentinel.
        if self._spec.g_callable is not None:
            return self._oracle_g_handle()
        return "raw_concat"

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> List[Optional[float]]:
        if self._spec.score_tree is None:
            raise RuntimeError(
                f"oracle {self._oracle_name!r} cannot score trees: spec has no "
                "score_tree adapter. Add one in the registration module so the "
                "alternating ladder can use this oracle as f."
            )
        results: List[Optional[float]] = []
        for tree in trees:
            try:
                value = self._spec.score_tree(tree)
            except Exception as exc:  # pragma: no cover - defensive logging surface
                results.append(None)
                continue
            try:
                results.append(float(value))
            except (TypeError, ValueError):
                # Vector-valued oracles (e.g. type_oracle) cannot be coerced
                # to a scalar root prediction; the caller is responsible for
                # using a compatible eval surface or supplying its own
                # adapter.
                results.append(None)
        return results

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        # The artifact returned by train_f/train_g is the oracle handle
        # (a string or None); nothing else to validate.
        if kind == "f" and artifact != self._oracle_handle():
            raise ValueError(
                f"oracle {self._oracle_name!r}: unexpected f artifact {artifact!r}"
            )
        if kind == "g" and self._spec.g_callable is not None:
            if artifact != self._oracle_g_handle():
                raise ValueError(
                    f"oracle {self._oracle_name!r}: unexpected g artifact {artifact!r}"
                )

    # --------------------- BundleAwareFamilyRuntime --------------------- #

    @property
    def default_f(self) -> str:
        return f"oracle:{self._oracle_name}"

    @property
    def default_g(self) -> str:
        if self._spec.g_callable is not None:
            return f"oracle:{self._oracle_name}"
        return "raw_concat"

    def expected_bundle(self) -> Mapping[str, Any]:
        return {
            "domain": self._spec.domain,
            "leaf_unit": self._spec.leaf_unit,
        }

    def supported_inits(self) -> Mapping[str, frozenset[str]]:
        f_set = frozenset({"oracle"})
        g_set = (
            frozenset({"oracle", "raw_concat"})
            if self._spec.g_callable is not None
            else frozenset({"raw_concat"})
        )
        return {"f": f_set, "g": g_set}

    def resolve_init(self, *, kind: str, spec: str) -> Any:
        # Accept either the bare oracle name or "oracle:<name>".
        text = str(spec).strip()
        if text.startswith("oracle:"):
            requested = text.split(":", 1)[1].strip()
        else:
            requested = text
        if str(kind) == "g" and requested == "raw_concat":
            return "raw_concat"
        if requested != self._oracle_name:
            raise ValueError(
                f"oracle family {self._oracle_name!r} cannot resolve "
                f"{kind}={spec!r}; expected oracle:{self._oracle_name}"
            )
        return self._oracle_handle() if kind == "f" else self._oracle_g_handle()

    def share_state_axes(self) -> frozenset[str]:
        return frozenset()

    # --------------------------- internals ---------------------------- #

    def _oracle_handle(self) -> str:
        return f"oracle:{self._oracle_name}"

    def _oracle_g_handle(self) -> str:
        return f"oracle:{self._oracle_name}:g"


__all__ = ["OracleFamilyRuntime"]
