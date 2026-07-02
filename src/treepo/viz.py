"""Standalone HTML visualization for tree records.

``write_tree_visualization_html`` renders trees as an expandable node tree in
one self-contained HTML file: no server, no JavaScript dependencies. Each node
shows its gold label, any prediction-style metadata, per-node ``f`` readouts,
sampling status (observed marker plus propensity/IPW weight), and local-law
losses, with text and full metadata behind a click. Audit summaries and error
certificates render as panels above the trees.

Inputs are the package's existing artifacts:

- trees: anything ``TreeRecord.from_value`` accepts;
- sampling rows: dicts shaped like ``manifesto_document_unit_sampling_rows``
  output (``tree_id``/``doc_id`` + ``node_id``/``unit_id`` + ``observed`` /
  propensity fields);
- law rows: ``LocalLawAuditRow`` objects or dicts whose metadata carries
  ``tree_id`` and ``node_id`` (the ``local_law_rows_from_tree_records`` shape),
  or rows with ``row_id`` shaped ``"<tree_id>:state:<node_index>"`` (the
  neural-operator statistic shape). Trace-indexed rows cover the family's
  pairwise merge schedule, so when a flat record (leaves plus root) carries
  them, the view synthesizes the intermediate merge nodes and shows losses on
  the actual computation tree.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from treepo.common import jsonable
from treepo.methods._coerce import safe_float
from treepo.methods._fno_transition import _pairwise_merge_children
from treepo.state import split_unit_id, state_to_dict
from treepo.tree import TreeRecord

DEFAULT_LABEL_KEYS = (
    "score",
    "llm_score",
    "llm_label",
    "prediction",
    "predicted",
    "predicted_label",
    "f_score",
)

DEFAULT_SUMMARY_KEYS = (
    "summary",
    "llm_summary",
    "g_summary",
    "state_text",
    "summary_text",
)


def tree_visualization_payload(
    tree: Any,
    *,
    sampling_rows: Iterable[Mapping[str, Any]] | None = None,
    law_rows: Iterable[Any] | None = None,
    readout_rows: Iterable[Mapping[str, Any]] | None = None,
    label_keys: Sequence[str] = DEFAULT_LABEL_KEYS,
    summary_keys: Sequence[str] = DEFAULT_SUMMARY_KEYS,
) -> dict[str, Any]:
    """Return one tree's JSONable visualization payload."""

    record = TreeRecord.from_value(tree)
    tree_ids = {str(record.tree_id), str(record.doc_id)}
    sampling_by_node = _rows_by_node(sampling_rows or (), tree_ids)
    laws_by_node, laws_by_trace_index = _index_law_rows(law_rows or (), tree_ids)
    readouts_by_node, readouts_by_trace = _index_readout_rows(readout_rows or (), tree_ids)

    by_id = {str(node.node_id): node for node in record.nodes}
    children: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for node in record.nodes:
        for child_id in (node.left_child_id, node.right_child_id):
            if child_id is not None and str(child_id) in by_id:
                children[str(node.node_id)].append(str(child_id))
    for node in record.nodes:
        if node.parent_id is not None and str(node.parent_id) in by_id:
            parent_children = children[str(node.parent_id)]
            if str(node.node_id) not in parent_children:
                parent_children.append(str(node.node_id))
    for node_id in children:
        children[node_id].sort(
            key=lambda cid: (
                by_id[cid].position if by_id[cid].position is not None else 0,
                cid,
            )
        )

    def node_payload(node_id: str, seen: set[str]) -> dict[str, Any]:
        node = by_id[node_id]
        seen.add(node_id)
        metadata = dict(node.metadata or {})
        labels = {
            key: jsonable(metadata[key])
            for key in label_keys
            if metadata.get(key) is not None
        }
        state = state_to_dict(node.state) if node.state is not None else None
        return {
            "node_id": node_id,
            "unit_type": str(node.unit_type),
            "text": str(node.text or ""),
            "level": node.level,
            "position": node.position,
            "label": jsonable(node.label),
            "labels": labels,
            "state": state,
            "summaries": _node_summaries(metadata, state, summary_keys),
            "metadata": jsonable(metadata),
            "sampling": sampling_by_node.get(node_id),
            "readout": readouts_by_node.get(node_id),
            "laws": laws_by_node.get(node_id, []),
            "children": [
                node_payload(child_id, seen)
                for child_id in children.get(node_id, [])
                if child_id not in seen
            ],
        }

    seen: set[str] = set()
    root = record.root()
    roots: list[dict[str, Any]] = []
    if root is not None:
        roots.append(node_payload(str(root.node_id), seen))
    for node in record.nodes:
        if str(node.node_id) not in seen:
            roots.append(node_payload(str(node.node_id), seen))

    if (laws_by_trace_index or readouts_by_trace) and roots:
        _attach_trace_annotations(roots[0], laws_by_trace_index, readouts_by_trace)

    meta = dict(record.metadata or {})
    n_sampled = sum(1 for row in sampling_by_node.values() if row.get("observed"))
    return {
        "tree_id": str(record.tree_id),
        "doc_id": record.doc_id,
        "root_label": jsonable(record.root_label),
        "n_nodes": len(record.nodes),
        "n_leaves": len(record.leaves()),
        "n_sampled_nodes": n_sampled,
        "document_observed": meta.get("document_observed"),
        "document_propensity": meta.get("document_propensity"),
        "metadata": jsonable(meta),
        "roots": roots,
    }


def write_tree_visualization_html(
    trees: Sequence[Any],
    path: Path | str,
    *,
    sampling_rows: Iterable[Mapping[str, Any]] | None = None,
    law_rows: Iterable[Any] | None = None,
    readout_rows: Iterable[Mapping[str, Any]] | None = None,
    audit: Mapping[str, Any] | None = None,
    certificate: Mapping[str, Any] | None = None,
    tradeoff: Mapping[str, Any] | None = None,
    label_keys: Sequence[str] = DEFAULT_LABEL_KEYS,
    summary_keys: Sequence[str] = DEFAULT_SUMMARY_KEYS,
    title: str = "treepo trees",
) -> Path:
    """Write a standalone expandable-tree HTML file and return its path.

    ``audit`` takes an ``audit_local_laws`` payload and renders it as a
    summary panel above the trees; ``certificate`` takes an error-certificate
    dict (``UnifiedLearningErrorCertificate.to_dict()``) and renders the
    component-radius ledger; ``tradeoff`` takes a ``TradeoffCurve.to_dict()``
    payload and renders the metric-vs-axis line chart with a table view.
    """

    tree_list = list(trees or ())
    # Group rows by their declared tree once, so each payload call only sees
    # its own tree's rows. Rows that name no tree apply to every tree only
    # when a single tree is rendered; in a batch they are ambiguous (generic
    # node ids like "root" repeat across trees) and are dropped.
    sampling_by_tree = _group_rows_by_tree(
        (row for row in sampling_rows or () if isinstance(row, Mapping)),
        tree_key=_sampling_row_tree,
    )
    laws_by_tree = _group_rows_by_tree(law_rows or (), tree_key=_law_row_tree)
    readouts_by_tree = _group_rows_by_tree(
        (row for row in readout_rows or () if isinstance(row, Mapping)),
        tree_key=_sampling_row_tree,
    )
    untagged_ok = len(tree_list) == 1
    tree_payloads = []
    for tree in tree_list:
        record_ids = _record_tree_ids(tree)
        tree_sampling = [row for key in record_ids for row in sampling_by_tree.get(key, [])]
        tree_laws = [row for key in record_ids for row in laws_by_tree.get(key, [])]
        tree_readouts = [row for key in record_ids for row in readouts_by_tree.get(key, [])]
        if untagged_ok:
            tree_sampling += sampling_by_tree.get("", [])
            tree_laws += laws_by_tree.get("", [])
            tree_readouts += readouts_by_tree.get("", [])
        tree_payloads.append(
            tree_visualization_payload(
                tree,
                sampling_rows=tree_sampling,
                law_rows=tree_laws,
                readout_rows=tree_readouts,
                label_keys=label_keys,
                summary_keys=summary_keys,
            )
        )
    payload = {
        "trees": tree_payloads,
        "audit": None if audit is None else jsonable(dict(audit)),
        "certificate": None if certificate is None else jsonable(dict(certificate)),
        "tradeoff": None if tradeoff is None else jsonable(dict(tradeoff)),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Escape "</" so node text containing "</script>" can't terminate the
    # embedded JSON script element; the payload stays valid JSON.
    encoded = json.dumps(jsonable(payload), sort_keys=True).replace("</", "<\\/")
    document = _HTML_TEMPLATE.replace("__TITLE__", html.escape(str(title))).replace(
        "__PAYLOAD__", encoded
    )
    out.write_text(document, encoding="utf-8")
    return out


def _rows_by_node(
    rows: Iterable[Mapping[str, Any]],
    tree_ids: set[str],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_tree = str(row.get("tree_id") or row.get("doc_id") or "")
        if row_tree and row_tree not in tree_ids:
            continue
        node_id = _row_node_id(row)
        if node_id is None:
            continue
        out[node_id] = {
            "observed": bool(row.get("observed")),
            "document_propensity": safe_float(row.get("document_propensity")),
            "unit_propensity": safe_float(row.get("unit_propensity")),
            "joint_propensity": safe_float(
                row.get("joint_propensity", row.get("inclusion_probability"))
            ),
            "ipw_weight": safe_float(row.get("ipw_weight")),
            "policy_name": row.get("policy_name"),
        }
    return out


def _law_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Project one law row onto the fields the HTML renderer reads."""

    law_kind = payload.get("law_kind")
    return {
        "law_kind": getattr(law_kind, "value", law_kind),
        "proxy_loss": safe_float(payload.get("proxy_loss")),
        "oracle_loss": safe_float(payload.get("oracle_loss")),
        "observed": bool(payload.get("observed")),
        "propensity": safe_float(payload.get("propensity")),
        "depth": payload.get("depth"),
    }


def _index_law_rows(
    rows: Iterable[Any],
    tree_ids: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """Index law rows by node id and, for statistic rows, by trace index.

    Node-keyed rows carry ``tree_id``/``node_id`` in their metadata (the
    ``local_law_rows_from_tree_records`` shape); trace-keyed rows carry a
    ``row_id`` shaped ``"<tree_id>:state:<node_index>"`` (the neural-operator
    statistic shape). Rows that declare a different tree are skipped; rows
    that declare no tree apply to the tree being rendered.
    """

    by_node: dict[str, list[dict[str, Any]]] = {}
    by_trace: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        payload = row.to_dict() if hasattr(row, "to_dict") else row
        if not isinstance(payload, Mapping):
            continue
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        row_tree = str(metadata.get("tree_id") or metadata.get("doc_id") or "")
        node_id = _row_node_id(metadata) or _row_node_id(payload)
        if node_id is not None:
            if not row_tree or row_tree in tree_ids:
                by_node.setdefault(node_id, []).append(_law_payload(payload))
            continue
        row_id = payload.get("row_id")
        if isinstance(row_id, str) and ":state:" in row_id:
            row_tree, _, index_text = row_id.rpartition(":state:")
            if row_tree in tree_ids and index_text.isdigit():
                by_trace.setdefault(int(index_text), []).append(_law_payload(payload))
    return by_node, by_trace


def _node_summaries(
    metadata: Mapping[str, Any],
    state: Mapping[str, Any] | None,
    summary_keys: Sequence[str],
) -> dict[str, str]:
    """Collect summary-style text produced for a node.

    Reads the configured metadata keys plus the ``text`` field of a
    ``TaskState`` (the summary ``g`` wrote for this node).
    """

    out: dict[str, str] = {}
    for key in summary_keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            out[key] = value
    if isinstance(state, Mapping):
        state_text = state.get("text")
        if isinstance(state_text, str) and state_text.strip():
            out.setdefault("state", state_text)
    return out


def _index_readout_rows(
    rows: Iterable[Mapping[str, Any]],
    tree_ids: set[str],
) -> tuple[dict[str, Any], dict[int, Any]]:
    """Index readout rows by node id and, for trace rows, by node index.

    Rows are mappings with a ``value`` plus either a node key (``node_id`` /
    ``unit_id``) or a ``node_index`` into the merge trace (the statistic's
    ``node_readouts`` shape).
    """

    by_node: dict[str, Any] = {}
    by_trace: dict[int, Any] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_tree = str(row.get("tree_id") or row.get("doc_id") or "")
        if row_tree and row_tree not in tree_ids:
            continue
        value = row.get("value")
        if value is None:
            continue
        node_id = _row_node_id(row)
        if node_id is not None:
            by_node[node_id] = jsonable(value)
        elif row.get("node_index") is not None:
            by_trace[int(row["node_index"])] = jsonable(value)
    return by_node, by_trace


def _record_tree_ids(tree: Any) -> tuple[str, ...]:
    record = TreeRecord.from_value(tree)
    ids = {str(record.tree_id), str(record.doc_id)}
    return tuple(sorted(ids))


def _sampling_row_tree(row: Any) -> str:
    if not isinstance(row, Mapping):
        return ""
    return str(row.get("tree_id") or row.get("doc_id") or "")


def _law_row_tree(row: Any) -> str:
    payload = row.to_dict() if hasattr(row, "to_dict") else row
    if not isinstance(payload, Mapping):
        return ""
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    declared = str(metadata.get("tree_id") or metadata.get("doc_id") or "")
    if declared:
        return declared
    row_id = payload.get("row_id")
    if isinstance(row_id, str) and ":state:" in row_id:
        return row_id.rpartition(":state:")[0]
    return ""


def _group_rows_by_tree(rows: Iterable[Any], *, tree_key: Any) -> dict[str, list[Any]]:
    out: dict[str, list[Any]] = {}
    for row in rows:
        out.setdefault(tree_key(row), []).append(row)
    return out


def _attach_trace_annotations(
    root_payload: dict[str, Any],
    laws_by_trace_index: Mapping[int, list[dict[str, Any]]],
    readouts_by_trace: Mapping[int, Any],
) -> None:
    """Attach trace-indexed law rows and readouts, synthesizing merge nodes.

    Trace indices follow the family schedule: leaves ``0..L-1`` in position
    order, then each merge level bottom-up, root state last (index ``2L-2``).
    Applies only when the record is a flat star — every non-root node is a
    childless child of the root — which is how task fixtures store trees.
    """

    children = root_payload.get("children") or []
    if any(child.get("children") for child in children):
        return
    leaf_count = len(children)
    max_index = max([*laws_by_trace_index, *readouts_by_trace])
    if max_index > max(0, 2 * leaf_count - 2):
        raise ValueError(
            f"trace rows reference node index {max_index}, but a tree with "
            f"{leaf_count} leaves has {max(1, 2 * leaf_count - 1)} trace nodes; "
            "the record's leaves and the scored tree's leaves disagree"
        )

    def annotate(node: dict[str, Any], index: int) -> None:
        node.setdefault("laws", []).extend(laws_by_trace_index.get(index, []))
        if node.get("readout") is None and index in readouts_by_trace:
            node["readout"] = readouts_by_trace[index]

    if leaf_count < 2:
        if children:
            annotate(children[0], 0)
        return

    for position, leaf in enumerate(children):
        annotate(leaf, position)

    merge_children = _pairwise_merge_children(leaf_count)
    root_index = 2 * leaf_count - 2

    payload_by_index: dict[int, dict[str, Any]] = dict(enumerate(children))

    def build(index: int) -> dict[str, Any]:
        if index in payload_by_index:
            return payload_by_index[index]
        left, right = merge_children[index]
        node = {
            "node_id": f"merge_{index}",
            "unit_type": "merge",
            "text": "",
            "level": None,
            "position": None,
            "label": None,
            "labels": {},
            "state": None,
            "summaries": {},
            "metadata": {"synthesized": True, "trace_index": index},
            "sampling": None,
            "readout": readouts_by_trace.get(index),
            "laws": list(laws_by_trace_index.get(index, [])),
            "children": [build(left), build(right)],
        }
        payload_by_index[index] = node
        return node

    left, right = merge_children[root_index]
    root_payload["children"] = [build(left), build(right)]
    annotate(root_payload, root_index)


def _row_node_id(row: Mapping[str, Any]) -> str | None:
    node_id = row.get("node_id")
    if node_id is not None:
        return str(node_id)
    unit_id = row.get("unit_id")
    if isinstance(unit_id, str) and ":" in unit_id:
        return split_unit_id(unit_id)[1]
    return None


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 1.5rem; color: #1a1a1a; }
  h1 { font-size: 1.2rem; }
  .legend { font-size: 0.8rem; color: #555; margin-bottom: 1rem; }
  .tree-card { border: 1px solid #ddd; border-radius: 6px; padding: 0.4rem 0.8rem; margin-bottom: 0.6rem; }
  .tree-card > summary { cursor: pointer; font-weight: 600; }
  .tree-stats { font-size: 0.8rem; color: #555; margin: 0.2rem 0 0.4rem; }
  details.node { margin-left: 1.1rem; border-left: 2px solid #eee; padding-left: 0.6rem; margin-top: 0.15rem; }
  details.node > summary { cursor: pointer; list-style: none; padding: 0.1rem 0; }
  details.node > summary::before { content: "\\25B8"; font-size: 0.7rem; color: #999; margin-right: 0.3rem; }
  details.node[open] > summary::before { content: "\\25BE"; }
  details.node.leaf > summary::before { content: "\\2022"; color: #bbb; }
  .chip { display: inline-block; font-size: 0.72rem; border-radius: 3px; padding: 0 0.35rem; margin-left: 0.3rem; }
  .chip.kind { background: #eef; color: #335; }
  .chip.gold { background: #e6f4e6; color: #1d5e1d; }
  .chip.pred { background: #e6eefb; color: #1c4587; }
  .chip.proxy { background: #fdf0e0; color: #8a5200; }
  .chip.oracle { background: #fbe8e8; color: #8f2727; }
  .chip.readout { background: #f0e6fb; color: #5e2a8a; }
  .panel { border: 1px solid #ddd; border-radius: 6px; padding: 0.5rem 0.8rem; margin-bottom: 0.8rem; font-size: 0.82rem; }
  .panel h2 { font-size: 0.9rem; margin: 0 0 0.3rem; }
  .panel table { border-collapse: collapse; }
  .panel td, .panel th { padding: 0.1rem 0.7rem 0.1rem 0; text-align: left; font-weight: normal; }
  .panel th { color: #555; }
  .panel .legend-row { font-size: 0.78rem; color: #0b0b0b; margin-bottom: 0.2rem; }
  .panel .legend-row .swatch { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 2px; margin: 0 0.25rem 0 0.7rem; vertical-align: baseline; }
  .panel .legend-row .swatch:first-child { margin-left: 0; }
  .dot { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 50%; margin-right: 0.3rem; vertical-align: baseline; }
  .dot.sampled { background: #2e7d32; }
  .dot.unsampled { background: #fff; border: 1px solid #999; }
  .node-text { font-size: 0.82rem; color: #333; white-space: pre-wrap; margin: 0.2rem 0; }
  .snippet { color: #777; font-style: italic; font-size: 0.78rem; margin-left: 0.45rem; }
  .summary-block { background: #f3f7f3; border-left: 3px solid #9c9; padding: 0.25rem 0.5rem; font-size: 0.8rem; margin: 0.2rem 0; white-space: pre-wrap; }
  .summary-block .tag { font-weight: 600; color: #2e7d32; margin-right: 0.4rem; font-style: normal; }
  .kv { font-size: 0.75rem; color: #555; margin: 0.1rem 0 0.3rem; }
  .kv code { background: #f6f6f6; padding: 0 0.2rem; }
  .dim details.node:not(.sampled-path) > summary { opacity: 0.35; }
  #controls { margin-bottom: 0.8rem; font-size: 0.85rem; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="legend">
  <span class="dot sampled"></span> sampled (observed) &nbsp;
  <span class="dot unsampled"></span> in population, unsampled &nbsp;
  <span class="chip gold">gold</span> label &nbsp;
  <span class="chip pred">prediction</span> metadata &nbsp;
  <span class="chip readout">f&#8594;</span> node readout &nbsp;
  <span class="chip proxy">proxy</span>/<span class="chip oracle">oracle</span> local-law loss
</div>
<div id="panels"></div>
<div id="controls">
  <label><input type="checkbox" id="dim-unsampled"> dim unsampled nodes</label>
  <button id="expand-all">expand all</button>
  <button id="collapse-all">collapse all</button>
</div>
<div id="trees"></div>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
(function () {
  const data = JSON.parse(document.getElementById("payload").textContent);
  const trees = data.trees;
  const container = document.getElementById("trees");
  const panels = document.getElementById("panels");

  function chip(cls, text) {
    const span = document.createElement("span");
    span.className = "chip " + cls;
    span.textContent = text;
    return span;
  }

  function fmt(value) {
    if (Array.isArray(value)) return "[" + value.map(fmt).join(", ") + "]";
    if (typeof value === "number" && !Number.isInteger(value)) return value.toPrecision(4);
    return String(value);
  }

  function panelTable(title, headers, rows) {
    const panel = document.createElement("div");
    panel.className = "panel";
    const heading = document.createElement("h2");
    heading.textContent = title;
    panel.appendChild(heading);
    const table = document.createElement("table");
    const headRow = document.createElement("tr");
    for (const header of headers) {
      const th = document.createElement("th");
      th.textContent = header;
      headRow.appendChild(th);
    }
    table.appendChild(headRow);
    for (const row of rows) {
      const tr = document.createElement("tr");
      for (const cell of row) {
        const td = document.createElement("td");
        td.textContent = cell;
        tr.appendChild(td);
      }
      table.appendChild(tr);
    }
    panel.appendChild(table);
    return panel;
  }

  if (data.audit) {
    const rows = [];
    const overall = data.audit.local_law_objective || {};
    const overlap = data.audit.influence_weighted_overlap || {};
    rows.push([
      "all laws",
      fmt(overall.objective), String(overall.objective_mode || ""),
      fmt(overall.row_count), fmt(overall.observed_count),
      fmt(overlap.effective_sample_size), fmt(overlap.max_weight),
    ]);
    for (const [kind, entry] of Object.entries(data.audit.by_law_kind || {})) {
      const obj = entry.local_law_objective || {};
      const ovl = entry.influence_weighted_overlap || {};
      rows.push([
        kind, fmt(obj.objective), String(obj.objective_mode || ""),
        fmt(obj.row_count), fmt(obj.observed_count),
        fmt(ovl.effective_sample_size), fmt(ovl.max_weight),
      ]);
    }
    panels.appendChild(panelTable(
      "Local-law audit",
      ["law", "objective", "mode", "rows", "observed", "ESS", "max weight"],
      rows,
    ));
  }

  if (data.certificate) {
    const cert = data.certificate;
    const rows = [
      ["reported estimate", fmt(cert.reported_estimate)],
      ["local-law radius", fmt(cert.local_law_radius)],
      ["calibration radius", fmt(cert.calibration_radius)],
      ["estimation radius", fmt(cert.estimation_radius)],
      ["clipping radius", fmt(cert.clipping_radius)],
      ["radius sum", fmt(cert.radius_sum)],
      ["total bound", fmt(cert.total_bound)],
    ];
    if (cert.confidence_delta !== null && cert.confidence_delta !== undefined) {
      rows.push(["confidence delta", fmt(cert.confidence_delta)]);
    }
    panels.appendChild(panelTable("Error certificate", ["component", "value"], rows));
  }

  // Validated categorical slots (light surface); assigned in fixed order.
  const SERIES_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#008300"];
  const SVG_NS = "http://www.w3.org/2000/svg";

  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  }

  function renderTradeoffPanel(curve) {
    const metricKeys = (curve.metric_keys || []).slice(0, SERIES_COLORS.length);
    const points = curve.points || [];
    const panel = document.createElement("div");
    panel.className = "panel";
    const heading = document.createElement("h2");
    heading.textContent = "Tradeoff: " +
      (metricKeys.length === 1 ? metricKeys[0] + " vs " : "") + curve.axis_kind;
    panel.appendChild(heading);

    if (metricKeys.length > 1) {
      const legend = document.createElement("div");
      legend.className = "legend-row";
      for (const [idx, key] of metricKeys.entries()) {
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = SERIES_COLORS[idx];
        legend.appendChild(swatch);
        legend.appendChild(document.createTextNode(key));
      }
      panel.appendChild(legend);
    }

    const width = 480, height = 190;
    const margin = {top: 10, right: 70, bottom: 28, left: 46};
    const plotW = width - margin.left - margin.right;
    const plotH = height - margin.top - margin.bottom;
    const xs = points.map((p) => p.axis_value);
    const allValues = points.flatMap((p) =>
      metricKeys.map((key) => p.metrics[key]).filter((v) => v !== null && v !== undefined));
    if (!xs.length || !allValues.length) return panel;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMax = Math.max(...allValues) || 1;
    const x = (v) => margin.left + (xMax === xMin ? plotW / 2 : ((v - xMin) / (xMax - xMin)) * plotW);
    const y = (v) => margin.top + plotH - (v / yMax) * plotH;

    const svg = svgEl("svg", {width, height, viewBox: `0 0 ${width} ${height}`, role: "img"});
    for (const frac of [0, 0.5, 1]) {
      const gy = margin.top + plotH - frac * plotH;
      svg.appendChild(svgEl("line", {x1: margin.left, y1: gy, x2: margin.left + plotW, y2: gy,
        stroke: "#e7e7e4", "stroke-width": 1}));
      const tick = svgEl("text", {x: margin.left - 6, y: gy + 3, "text-anchor": "end",
        "font-size": 10, fill: "#52514e"});
      tick.textContent = fmt(frac * yMax);
      svg.appendChild(tick);
    }
    for (const xv of xs) {
      const tick = svgEl("text", {x: x(xv), y: height - 10, "text-anchor": "middle",
        "font-size": 10, fill: "#52514e"});
      tick.textContent = fmt(xv);
      svg.appendChild(tick);
    }
    const axisTitle = svgEl("text", {x: margin.left + plotW / 2, y: height - 1,
      "text-anchor": "middle", "font-size": 10, fill: "#52514e"});
    axisTitle.textContent = curve.axis_kind;
    svg.appendChild(axisTitle);

    for (const [idx, key] of metricKeys.entries()) {
      const color = SERIES_COLORS[idx];
      const series = points.filter((p) => p.metrics[key] !== null && p.metrics[key] !== undefined);
      if (!series.length) continue;
      const path = series.map((p, i) =>
        (i ? "L" : "M") + x(p.axis_value) + " " + y(p.metrics[key])).join(" ");
      svg.appendChild(svgEl("path", {d: path, fill: "none", stroke: color, "stroke-width": 2}));
      for (const p of series) {
        const marker = svgEl("circle", {cx: x(p.axis_value), cy: y(p.metrics[key]), r: 4,
          fill: color, stroke: "#ffffff", "stroke-width": 2});
        const tip = document.createElementNS(SVG_NS, "title");
        tip.textContent = `${curve.axis_kind} ${fmt(p.axis_value)} · ${key} ${fmt(p.metrics[key])}`;
        marker.appendChild(tip);
        svg.appendChild(marker);
      }
      const last = series[series.length - 1];
      const label = svgEl("text", {x: x(last.axis_value) + 8, y: y(last.metrics[key]) + 3,
        "font-size": 10, fill: "#0b0b0b"});
      label.textContent = fmt(last.metrics[key]);
      svg.appendChild(label);
    }
    panel.appendChild(svg);

    // The chart caps at four color slots; the table always shows every metric.
    const allKeys = curve.metric_keys || [];
    const table = panelTable(
      "",
      [curve.axis_kind, ...allKeys],
      points.map((p) => [fmt(p.axis_value), ...allKeys.map((key) => fmt(p.metrics[key]))]),
    );
    table.classList.remove("panel");
    table.querySelector("h2").remove();
    panel.appendChild(table);
    return panel;
  }

  if (data.tradeoff && (data.tradeoff.points || []).length) {
    panels.appendChild(renderTradeoffPanel(data.tradeoff));
  }

  function renderNode(node) {
    const details = document.createElement("details");
    details.className = "node" + (node.children.length ? "" : " leaf");
    const summary = document.createElement("summary");

    if (node.sampling) {
      const dot = document.createElement("span");
      dot.className = "dot " + (node.sampling.observed ? "sampled" : "unsampled");
      dot.title = node.sampling.observed ? "sampled" : "unsampled";
      summary.appendChild(dot);
      if (node.sampling.observed) details.classList.add("sampled-path");
    }
    summary.appendChild(document.createTextNode(node.node_id));
    summary.appendChild(chip("kind", node.unit_type));
    if (node.label !== null && node.label !== undefined) {
      summary.appendChild(chip("gold", "gold " + fmt(node.label)));
    }
    for (const [key, value] of Object.entries(node.labels || {})) {
      summary.appendChild(chip("pred", key + " " + fmt(value)));
    }
    if (node.readout !== null && node.readout !== undefined) {
      summary.appendChild(chip("readout", "f\\u2192 " + fmt(node.readout)));
    }
    for (const law of node.laws || []) {
      if (law.proxy_loss !== null && law.proxy_loss !== undefined) {
        summary.appendChild(chip("proxy", (law.law_kind || "law") + " " + fmt(law.proxy_loss)));
      }
      if (law.oracle_loss !== null && law.oracle_loss !== undefined && law.observed) {
        summary.appendChild(chip("oracle", "oracle " + fmt(law.oracle_loss)));
      }
    }
    const summaries = node.summaries || {};
    const snippetSource = node.text || Object.values(summaries)[0] || "";
    if (snippetSource) {
      const snippet = document.createElement("span");
      snippet.className = "snippet";
      snippet.textContent =
        snippetSource.length > 90 ? snippetSource.slice(0, 90) + "\\u2026" : snippetSource;
      summary.appendChild(snippet);
    }
    details.appendChild(summary);

    if (node.text) {
      const text = document.createElement("div");
      text.className = "node-text";
      text.textContent = node.text;
      details.appendChild(text);
    }
    for (const [key, value] of Object.entries(summaries)) {
      const block = document.createElement("div");
      block.className = "summary-block";
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = key;
      block.appendChild(tag);
      block.appendChild(document.createTextNode(value));
      details.appendChild(block);
    }
    const kv = document.createElement("div");
    kv.className = "kv";
    const facts = [];
    if (node.sampling) {
      if (node.sampling.joint_propensity != null) facts.push("propensity " + fmt(node.sampling.joint_propensity));
      if (node.sampling.ipw_weight != null) facts.push("ipw " + fmt(node.sampling.ipw_weight));
    }
    for (const law of node.laws || []) {
      if (law.depth != null) facts.push("depth " + law.depth);
    }
    if (facts.length) {
      kv.appendChild(document.createTextNode(facts.join(" · ") + " "));
    }
    const metaCode = document.createElement("code");
    metaCode.textContent = JSON.stringify(node.metadata);
    const metaWrap = document.createElement("details");
    const metaSummary = document.createElement("summary");
    metaSummary.textContent = "metadata";
    metaWrap.appendChild(metaSummary);
    metaWrap.appendChild(metaCode);
    if (node.state) {
      const stateCode = document.createElement("code");
      stateCode.textContent = JSON.stringify(node.state);
      const stateLabel = document.createElement("div");
      stateLabel.textContent = "state:";
      metaWrap.appendChild(stateLabel);
      metaWrap.appendChild(stateCode);
    }
    kv.appendChild(metaWrap);
    details.appendChild(kv);

    for (const child of node.children) details.appendChild(renderNode(child));
    return details;
  }

  for (const tree of trees) {
    const card = document.createElement("details");
    card.className = "tree-card";
    card.open = trees.length === 1;
    const summary = document.createElement("summary");
    summary.textContent = tree.tree_id;
    if (tree.root_label !== null && tree.root_label !== undefined) {
      summary.appendChild(chip("gold", "root " + fmt(tree.root_label)));
    }
    if (tree.document_observed != null) {
      summary.appendChild(chip("kind", tree.document_observed ? "doc sampled" : "doc unsampled"));
    }
    card.appendChild(summary);
    const stats = document.createElement("div");
    stats.className = "tree-stats";
    const parts = [tree.n_nodes + " nodes", tree.n_leaves + " leaves"];
    if (tree.n_sampled_nodes) parts.push(tree.n_sampled_nodes + " sampled");
    if (tree.document_propensity != null) parts.push("doc propensity " + fmt(tree.document_propensity));
    stats.textContent = parts.join(" · ");
    card.appendChild(stats);
    for (const root of tree.roots) card.appendChild(renderNode(root));
    container.appendChild(card);
  }

  document.getElementById("dim-unsampled").addEventListener("change", (event) => {
    container.classList.toggle("dim", event.target.checked);
  });
  document.getElementById("expand-all").addEventListener("click", () => {
    container.querySelectorAll("details").forEach((d) => (d.open = true));
  });
  document.getElementById("collapse-all").addEventListener("click", () => {
    container.querySelectorAll("details").forEach((d) => (d.open = false));
  });
})();
</script>
</body>
</html>
"""

__all__ = [
    "DEFAULT_LABEL_KEYS",
    "DEFAULT_SUMMARY_KEYS",
    "tree_visualization_payload",
    "write_tree_visualization_html",
]
