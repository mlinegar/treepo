"""Shared f/g arity and token-budget contract.

Canonical rule:
- f consumes one leaf-sized state: ``leaf_size_tokens``.
- g consumes two child states: ``2 * leaf_size_tokens``.
- g may emit their verbatim concatenation: ``2 * leaf_size_tokens``.

This module keeps that rule out of individual backends so DSPy, TRL, teacher
trace generation, and neural/operator backends fail consistently.

The same 2x rule is generalized via ``check_state_summary_invariant`` for
embedding-state (FNO) and sketch-state (HLL/CTreePO) families: every family
must guarantee ``state_dim >= 2 * summary_dim`` so that pure raw concatenation
of two children can always be represented in the parent state.

The single principled exception is lossy native sketch merges (e.g. HLL's
register-wise max). Those families pass ``allow_lossy_native=True`` AND must
declare a ``ConcatSketch``-equivalent default ``g`` that satisfies the strict
invariant so that ``raw_concat`` is still always available as a no-op fallback.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FGArityBudget:
    leaf_size_tokens: int
    f_input_tokens: int
    g_input_tokens: int
    g_output_tokens: int


@dataclass(frozen=True)
class FGLMBudgetReport:
    """Programmatic preflight report for the two-child f/g LM contract."""

    family_name: str
    leaf_size_tokens: int
    requested_max_completion_tokens: int | None
    max_completion_tokens: int
    lm_context_window_tokens: int
    prompt_template_overhead_tokens: int
    required_g_input_tokens: int
    required_g_output_tokens: int
    available_input_tokens: int
    minimum_context_window_tokens: int
    ok: bool
    violations: tuple[str, ...]

    def raise_for_error(self) -> None:
        if self.ok:
            return
        raise RuntimeError("; ".join(self.violations))


def fg_arity_budget(leaf_size_tokens: int) -> FGArityBudget:
    leaf = int(leaf_size_tokens)
    if leaf <= 0:
        raise ValueError(f"leaf_size_tokens must be positive, got {leaf_size_tokens}")
    return FGArityBudget(
        leaf_size_tokens=leaf,
        f_input_tokens=leaf,
        g_input_tokens=2 * leaf,
        g_output_tokens=2 * leaf,
    )


def auto_g_output_tokens(
    requested: int | None,
    *,
    leaf_size_tokens: int,
) -> int:
    required = fg_arity_budget(int(leaf_size_tokens)).g_output_tokens
    if requested is None or int(requested) <= 0:
        return int(required)
    if int(requested) < int(required):
        raise RuntimeError(
            f"g output budget too small: requested max tokens={requested}, "
            f"but 2 * leaf_size_tokens = {required}. g must be able to emit "
            "a verbatim concatenation of two children."
        )
    return int(requested)


def two_child_lm_budget_report(
    *,
    family_name: str,
    leaf_size_tokens: int,
    lm_context_window_tokens: int,
    max_completion_tokens: int | None,
    prompt_template_overhead_tokens: int,
) -> FGLMBudgetReport:
    """Return a structured fit report for the canonical two-child g contract.

    ``max_completion_tokens <= 0`` means automatic completion length equal to
    ``2 * leaf_size_tokens``.
    """
    budget = fg_arity_budget(int(leaf_size_tokens))
    requested = None if max_completion_tokens is None else int(max_completion_tokens)
    effective_completion = (
        int(budget.g_output_tokens)
        if requested is None or requested <= 0
        else int(requested)
    )
    context = int(lm_context_window_tokens)
    overhead = int(prompt_template_overhead_tokens)
    available_input = context - effective_completion - overhead
    minimum_context = int(budget.g_input_tokens) + int(budget.g_output_tokens) + overhead

    violations: list[str] = []
    if effective_completion < int(budget.g_output_tokens):
        violations.append(
            f"{family_name}: max_completion_tokens={effective_completion} "
            f"< 2 * leaf_size_tokens = {budget.g_output_tokens}. g must be able "
            "to emit a verbatim concatenation of two children."
        )
    if int(budget.g_input_tokens) > int(available_input):
        violations.append(
            f"{family_name}: two-child input budget exceeded. "
            f"2 * leaf_size_tokens = {budget.g_input_tokens}, available input "
            f"budget = {available_input} (= lm_context_window_tokens="
            f"{context} - max_completion_tokens={effective_completion} "
            f"- prompt_template_overhead_tokens={overhead})."
        )

    return FGLMBudgetReport(
        family_name=str(family_name),
        leaf_size_tokens=int(leaf_size_tokens),
        requested_max_completion_tokens=requested,
        max_completion_tokens=int(effective_completion),
        lm_context_window_tokens=int(context),
        prompt_template_overhead_tokens=int(overhead),
        required_g_input_tokens=int(budget.g_input_tokens),
        required_g_output_tokens=int(budget.g_output_tokens),
        available_input_tokens=int(available_input),
        minimum_context_window_tokens=int(minimum_context),
        ok=not violations,
        violations=tuple(violations),
    )


def check_two_child_lm_budget(
    *,
    family_name: str,
    leaf_size_tokens: int,
    lm_context_window_tokens: int,
    max_completion_tokens: int,
    prompt_template_overhead_tokens: int,
) -> None:
    two_child_lm_budget_report(
        family_name=family_name,
        leaf_size_tokens=int(leaf_size_tokens),
        lm_context_window_tokens=int(lm_context_window_tokens),
        max_completion_tokens=int(max_completion_tokens),
        prompt_template_overhead_tokens=int(prompt_template_overhead_tokens),
    ).raise_for_error()


# ---------------------------------------------------------------------------
# Generalized state/summary contract for non-LM families.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateSummaryReport:
    """Programmatic preflight report for the generalized 2x rule.

    Used by embedding (FNO) and sketch (HLL/CTreePO) families. The rule:
    ``state_dim >= 2 * summary_dim`` so that pure raw concatenation of two
    children always fits in the parent state.
    """

    family_name: str
    state_kind: str
    state_dim: int
    summary_dim: int
    g_in_dim: int
    g_out_dim: int
    allow_lossy_native: bool
    ok: bool
    violations: tuple[str, ...]

    def raise_for_error(self) -> None:
        if self.ok:
            return
        raise RuntimeError("; ".join(self.violations))


def state_summary_report(
    *,
    family_name: str,
    state_kind: str,
    state_dim: int,
    summary_dim: int,
    g_in_dim: int | None = None,
    g_out_dim: int | None = None,
    allow_lossy_native: bool = False,
) -> StateSummaryReport:
    """Build a structured report for the generalized state/summary contract.

    ``state_kind`` must be one of ``"embedding"``, ``"sketch_state"``, or
    ``"sketch_state_lossy_native"``. ``g_in_dim`` defaults to ``2*summary_dim``;
    ``g_out_dim`` defaults to ``state_dim``.
    """
    valid_kinds = {"embedding", "sketch_state", "sketch_state_lossy_native"}
    if state_kind not in valid_kinds:
        raise ValueError(
            f"state_kind must be one of {sorted(valid_kinds)}, got {state_kind!r}"
        )
    s_dim = int(state_dim)
    sum_dim = int(summary_dim)
    g_in = int(g_in_dim) if g_in_dim is not None else 2 * sum_dim
    g_out = int(g_out_dim) if g_out_dim is not None else s_dim

    violations: list[str] = []
    if sum_dim <= 0:
        violations.append(
            f"{family_name}: summary_dim must be positive, got {sum_dim}"
        )
    if s_dim <= 0:
        violations.append(
            f"{family_name}: state_dim must be positive, got {s_dim}"
        )
    if state_kind != "sketch_state_lossy_native":
        if s_dim < 2 * sum_dim:
            violations.append(
                f"{family_name}: state_dim={s_dim} < 2 * summary_dim = {2 * sum_dim}. "
                "Parent state must accommodate verbatim concatenation of two children."
            )
        if g_in < 2 * sum_dim:
            violations.append(
                f"{family_name}: g_in_dim={g_in} < 2 * summary_dim = {2 * sum_dim}. "
                "g must consume both child summaries together."
            )
        if g_out < 2 * sum_dim:
            violations.append(
                f"{family_name}: g_out_dim={g_out} < 2 * summary_dim = {2 * sum_dim}. "
                "g must be able to emit a verbatim concatenation of two children."
            )
    elif not allow_lossy_native:
        violations.append(
            f"{family_name}: state_kind=sketch_state_lossy_native requires "
            "allow_lossy_native=True. The lossy variant must be opted into "
            "explicitly and provide a ConcatSketch-equivalent default g."
        )

    return StateSummaryReport(
        family_name=str(family_name),
        state_kind=str(state_kind),
        state_dim=s_dim,
        summary_dim=sum_dim,
        g_in_dim=g_in,
        g_out_dim=g_out,
        allow_lossy_native=bool(allow_lossy_native),
        ok=not violations,
        violations=tuple(violations),
    )


def check_state_summary_invariant(
    *,
    family_name: str,
    state_kind: str,
    state_dim: int,
    summary_dim: int,
    g_in_dim: int | None = None,
    g_out_dim: int | None = None,
    allow_lossy_native: bool = False,
) -> None:
    """Raise if the generalized 2x state/summary rule is violated.

    Lossy-native sketch families (e.g. HLL register-wise max) must pass
    ``state_kind="sketch_state_lossy_native"`` and ``allow_lossy_native=True``;
    they are responsible for offering a ``ConcatSketch``-equivalent default g
    so ``raw_concat`` remains a no-op fallback.
    """
    state_summary_report(
        family_name=family_name,
        state_kind=state_kind,
        state_dim=state_dim,
        summary_dim=summary_dim,
        g_in_dim=g_in_dim,
        g_out_dim=g_out_dim,
        allow_lossy_native=allow_lossy_native,
    ).raise_for_error()


def check_two_child_embedding_budget(
    *,
    family_name: str,
    summary_dim: int,
    g_in_dim: int,
    g_out_dim: int,
    state_dim: int | None = None,
) -> None:
    """Embedding-axis specialization of the generalized 2x rule.

    ``state_dim`` defaults to ``g_out_dim`` (the parent state is whatever g
    produces). Use this from FNO/embedding-tree family ``__init__`` paths.
    """
    check_state_summary_invariant(
        family_name=family_name,
        state_kind="embedding",
        state_dim=int(state_dim) if state_dim is not None else int(g_out_dim),
        summary_dim=int(summary_dim),
        g_in_dim=int(g_in_dim),
        g_out_dim=int(g_out_dim),
    )


def check_two_child_sketch_budget(
    *,
    family_name: str,
    summary_units: int,
    g_in_units: int,
    g_out_units: int,
    state_units: int | None = None,
    allow_lossy_native: bool = False,
) -> None:
    """Sketch-state specialization of the generalized 2x rule.

    Pass ``allow_lossy_native=True`` for HLL-style register-wise max merges.
    Those callers must provide a ConcatSketch wrapper as the family's
    canonical default g; this check covers only the lossy named override.
    """
    state_kind = "sketch_state_lossy_native" if allow_lossy_native else "sketch_state"
    check_state_summary_invariant(
        family_name=family_name,
        state_kind=state_kind,
        state_dim=int(state_units) if state_units is not None else int(g_out_units),
        summary_dim=int(summary_units),
        g_in_dim=int(g_in_units),
        g_out_dim=int(g_out_units),
        allow_lossy_native=allow_lossy_native,
    )
