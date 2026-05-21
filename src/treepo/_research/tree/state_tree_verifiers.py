"""Verification hooks for stateful TreePO runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol

from treepo._research.core.async_utils import to_thread
from treepo._research.core.ops_checks import CheckResult, CheckType, LawKind
from treepo._research.tree.state_tree import StateNode, StateTree


def _lawkind_to_check_type(law: LawKind) -> CheckType:
    if law is LawKind.L1_LEAF:
        return CheckType.SUFFICIENCY
    if law is LawKind.L2_MERGE:
        return CheckType.MERGE
    if law is LawKind.L3_IDEMPOTENCE:
        return CheckType.IDEMPOTENCE
    raise ValueError(f"Unsupported LawKind: {law}")


class LawVerifier(Protocol):
    """Hook API for verifying local laws during a state-tree run."""

    name: str

    async def check_leaf(
        self,
        node: StateNode[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **kwargs: Any,
    ) -> Mapping[LawKind, CheckResult] | None:
        ...

    async def check_merge(
        self,
        node: StateNode[Any, Any],
        left: StateNode[Any, Any],
        right: StateNode[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **kwargs: Any,
    ) -> Mapping[LawKind, CheckResult] | None:
        ...

    async def check_idempotence(
        self,
        node: StateNode[Any, Any],
        *,
        prior_rendered: str,
        rubric: str,
        operator: Optional[Any] = None,
        **kwargs: Any,
    ) -> Mapping[LawKind, CheckResult] | None:
        ...

    async def finalize_tree(
        self,
        tree: StateTree[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **kwargs: Any,
    ) -> Mapping[str, Mapping[LawKind, CheckResult]] | None:
        ...


def _attach_law_checks(node: StateNode[Any, Any], checks: Mapping[LawKind, CheckResult], *, verifier_name: str) -> None:
    """Store law check results in the node audit slot."""
    slot = node.audit.setdefault("law_checks", {})
    per_verifier = slot.setdefault(str(verifier_name), {})
    for law, result in checks.items():
        per_verifier[law.value] = result.to_dict()


@dataclass(frozen=True)
class MarkovExactVerifier:
    """Exact verifier for the Markov toy sketch operator."""

    name: str = "markov_exact"

    async def check_leaf(
        self,
        node: StateNode[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **_: Any,
    ) -> Mapping[LawKind, CheckResult]:
        from treepo._research.diffusion.markov_toy import encode_markov_path

        span = list(node.span or [])
        expected = encode_markov_path(span)
        passed = node.state == expected
        return {
            LawKind.L1_LEAF: CheckResult(
                check_type=_lawkind_to_check_type(LawKind.L1_LEAF),
                passed=bool(passed),
                discrepancy=0.0 if passed else 1.0,
                reasoning="Exact leaf sketch match." if passed else "Leaf sketch mismatch against oracle.",
                node_id=node.id,
                skipped=False,
            )
        }

    async def check_merge(
        self,
        node: StateNode[Any, Any],
        left: StateNode[Any, Any],
        right: StateNode[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **_: Any,
    ) -> Mapping[LawKind, CheckResult]:
        from treepo._research.diffusion.markov_toy import encode_markov_path

        span = list(node.span or [])
        expected = encode_markov_path(span)
        passed = node.state == expected
        return {
            LawKind.L2_MERGE: CheckResult(
                check_type=_lawkind_to_check_type(LawKind.L2_MERGE),
                passed=bool(passed),
                discrepancy=0.0 if passed else 1.0,
                reasoning="Exact merge sketch match." if passed else "Merge sketch mismatch against oracle.",
                node_id=node.id,
                skipped=False,
            )
        }

    async def check_idempotence(
        self,
        node: StateNode[Any, Any],
        *,
        prior_rendered: str,
        rubric: str,
        operator: Optional[Any] = None,
        **_: Any,
    ) -> Mapping[LawKind, CheckResult]:
        # MarkovToySketch is not decodable; treat idempotence as not applicable.
        return {
            LawKind.L3_IDEMPOTENCE: CheckResult(
                check_type=_lawkind_to_check_type(LawKind.L3_IDEMPOTENCE),
                passed=False,
                discrepancy=0.0,
                reasoning="Skipped: Markov sketch states do not support decode/re-encode idempotence checks.",
                node_id=node.id,
                skipped=True,
                skip_reason="no_decode",
            )
        }

    async def finalize_tree(
        self,
        tree: StateTree[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        **_: Any,
    ) -> Mapping[str, Mapping[LawKind, CheckResult]]:
        # Nodewise verifier; no finalize step.
        return {}


@dataclass
class TextAuditorAdapterVerifier:
    """Audit a text ``StateTree`` using the legacy-auditor semantics.

    Unlike the first-pass adapter, this runs the auditor logic directly on
    ``StateTree[str, str]`` without converting into the legacy ``Tree`` type.
    """

    oracle: Any
    audit_config: Optional[Any] = None
    summarizer: Optional[Any] = None
    theorem_operator: Optional[Any] = None
    name: str = "text_auditor_adapter"

    async def finalize_tree(
        self,
        tree: StateTree[Any, Any],
        *,
        rubric: str,
        operator: Optional[Any] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **_: Any,
    ) -> Mapping[str, Mapping[LawKind, CheckResult]]:
        from treepo._research.tree.auditor import AuditConfig
        from treepo._research.tree.state_tree_auditor import StateTreeAuditor

        config = self.audit_config or AuditConfig()
        auditor = StateTreeAuditor(
            self.oracle,
            config=config,
            summarizer=self.summarizer,
            theorem_operator=self.theorem_operator,
        )
        report = await to_thread(
            auditor.audit_tree,
            tree,  # type: ignore[arg-type]
            rubric=str(rubric or ""),
        )
        # Preserve a compact audit report for debugging and downstream analysis.
        try:
            tree.metadata.setdefault("text_audit_report", report.to_dict())
        except Exception:
            pass

        results: Dict[str, Dict[LawKind, CheckResult]] = {}
        for check in list(getattr(report, "checks", []) or []):
            check_type = str(getattr(check, "check_type", "") or "")
            node_id = str(getattr(check, "node_id", "") or "")
            if not node_id:
                continue
            if check_type == "sufficiency":
                law = LawKind.L1_LEAF
            elif check_type == "merge_consistency":
                law = LawKind.L2_MERGE
            elif check_type == "idempotence":
                law = LawKind.L3_IDEMPOTENCE
            else:
                continue

            results.setdefault(node_id, {})[law] = CheckResult(
                check_type=_lawkind_to_check_type(law),
                passed=bool(getattr(check, "passed", False)),
                discrepancy=float(getattr(check, "discrepancy_score", 0.0) or 0.0),
                reasoning=str(getattr(check, "reasoning", "") or ""),
                input_a=str(getattr(check, "input_a", "") or ""),
                input_b=str(getattr(check, "input_b", "") or ""),
                node_id=node_id,
                skipped=bool(getattr(check, "skipped", False)),
                skip_reason=getattr(check, "skip_reason", None),
            )

        return results


__all__ = [
    "LawVerifier",
    "MarkovExactVerifier",
    "TextAuditorAdapterVerifier",
    "_attach_law_checks",
]
