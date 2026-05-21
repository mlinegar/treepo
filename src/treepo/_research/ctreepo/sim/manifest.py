from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from treepo._research.ctreepo.sim.execution_resources import infer_run_resources


def _stable_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _short_sha256(text: str, *, n: int = 16) -> str:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return h[: int(max(8, min(64, n)))]


@dataclass(frozen=True)
class RunSpec:
    """A single runnable simulation/plot/report command with reproducible metadata."""

    id: str
    family: str
    config: Dict[str, Any]
    outputs: Dict[str, str]
    command: str
    requires: List[str]
    resources: Dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        family: str,
        config: Dict[str, Any],
        outputs: Dict[str, str],
        command: str,
        requires: Sequence[str] | None = None,
        resources: Dict[str, Any] | None = None,
        id_len: int = 16,
    ) -> "RunSpec":
        payload = {"family": str(family), "config": dict(config), "outputs": dict(outputs)}
        rid = _short_sha256(_stable_dumps(payload), n=int(id_len))
        reqs = [str(x) for x in (list(requires) if requires is not None else [])]
        resource_payload = (
            dict(resources)
            if resources is not None
            else infer_run_resources(
                family=str(family),
                config=dict(config),
                requires=reqs,
                command=str(command),
            )
        )
        return cls(
            id=rid,
            family=str(family),
            config=dict(config),
            outputs={str(k): str(v) for k, v in dict(outputs).items()},
            command=str(command),
            requires=reqs,
            resources=resource_payload,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "problem_id": str(self.family),
            "method_id": str(dict(self.config).get("method_id", "simulation")),
            "config": dict(self.config),
            "outputs": dict(self.outputs),
            "command": str(self.command),
            "requires": list(self.requires),
            "resources": dict(self.resources),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunSpec":
        family = str(d.get("problem_id", d.get("family", "")))
        return cls(
            id=str(d.get("id", "")),
            family=family,
            config=dict(d.get("config", {}) or {}),
            outputs={str(k): str(v) for k, v in dict(d.get("outputs", {}) or {}).items()},
            command=str(d.get("command", "")),
            requires=[str(x) for x in (d.get("requires", []) or [])],
            resources=(
                dict(d.get("resources", {}) or {})
                or infer_run_resources(
                    family=family,
                    config=dict(d.get("config", {}) or {}),
                    requires=[str(x) for x in (d.get("requires", []) or [])],
                    command=str(d.get("command", "")),
                )
            ),
        )


def write_manifest_jsonl(path: Path, runs: Iterable[RunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_stable_dumps(r.to_dict()) for r in runs]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_manifest_jsonl(path: Path) -> List[RunSpec]:
    if not path.exists():
        return []
    runs: List[RunSpec] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        runs.append(RunSpec.from_dict(json.loads(line)))
    return runs


def existing_outputs(run: RunSpec, *, key: str = "json_summary") -> bool:
    out = dict(run.outputs)
    p = out.get(str(key), "")
    if not p:
        return False
    return Path(p).exists()
