# Labeled-tree bundle contract

`treepo.bundles.load_labeled_tree_bundle` reads externally produced labeled
trees into the package-owned `treepo.tree.TreeRecord` shape, so real supervision
grids run through `fit()` without porting any task into treepo. Producers own the
build step; treepo owns only the read side and stays task-agnostic. This page
pins the on-disk contract field by field.

## Layout

A bundle is a directory. Two shapes are supported:

- **A single leaf-scale run directory** containing `labeled_trees.jsonl` (one
  document tree per line) and, optionally, a sibling or parent `split_ids.json`.
- **A top-level grid directory** containing `manifest.json` and one or more
  leaf-scale run subdirectories (each with its own `labeled_trees.jsonl`), plus a
  bundle-level `split_ids.json`. When exactly one run subdirectory is present the
  loader resolves it automatically; when several are present the loader refuses
  to guess and names the run directories to pass instead (each leaf scale is a
  different topology and must be loaded explicitly).

`load_labeled_tree_bundle(path, *, split=None, dimension=None)` accepts a
`labeled_trees.jsonl` file, a run directory, or a single-run top-level directory.

## `labeled_trees.jsonl` — one tree per line

Each line is a JSON object.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `doc_id` | string | yes | Stable document identifier; becomes `tree_id` and `doc_id`. |
| `nodes` | object | yes | Map of `node_id -> node` (see node table). |
| `version` | string | no | Schema version (e.g. `"3.0"`). Preserved as `metadata["schema_version"]`. Unknown versions are tolerated. |
| `document_text` | string | no | Full document text; becomes the record `text`. |
| `document_score` | number | no | Tree-level target for the default dimension; becomes `root_label` when no `dimension` is selected. |
| `levels` | array of arrays | no | Node ids grouped by level, leaves first. Fixes node `position` within each level. |
| `num_chunks`, `num_levels` | int | no | Convenience counts; ignored by the loader. |
| `metadata` | object | no | Carried through onto the record `metadata`. Commonly holds `split` (`"train"`/`"val"`/`"test"`), `artifact_version`, and topology policy. |
| `created_at`, `label_source` | string | no | Provenance; `label_source` is surfaced on record metadata. |

### Node object

| Field | Type | Required | Meaning |
|---|---|---|---|
| `node_id` | string | yes | Unique within the tree. |
| `level` | int | yes | Tree depth; `0` = leaf, `1+` = merge node. Root is the deepest unparented node. |
| `text` | string | no | Node content. |
| `score` | number | no | Scalar node label for the default dimension; becomes the node `label`. |
| `dimension_scores` | object | no | Per-dimension labels (e.g. `rile`, `domain_1..domain_7`); selected by the `dimension` argument. |
| `left_child_id`, `right_child_id` | string / null | no | Child pointers; parent edges are reconstructed from these. |
| `reasoning`, `confidence`, `timestamp` | any | no | Provenance; preserved on node metadata. |
| `metadata` | object | no | Per-node payload — see below. |

### Node `metadata` payload

Every key here is preserved verbatim on the loaded node's `metadata`. Typical
keys from the Manifesto/RILE producers:

- `char_start`, `char_end` — span into `document_text`.
- `is_leaf`, `g_training_role` (`"leaf"` / `"merge"`).
- `cmp_counts` — per-code counts (the 56-code substrate, sparse).
- `domain_counts` — per-domain counts.
- `rile_raw`, `rile_norm`, `total_qsentences`, `total_non_header_qsentences`.
- `sentence_start_index`, `sentence_end_index`, `qsentence_start_index`,
  `qsentence_end_index`, `total_sentences` — index ranges into the document.
- `leaf_unit_count`, `leaf_sentences`, `topology_axis` — topology descriptors.
- teacher-summary fields (`teacher_summary`, `target_summary`, ...) when present.

The loader does not interpret these. It lifts the node-level typed fields
(`score`, `dimension_scores`, `reasoning`, `confidence`, `timestamp`, `doc_id`,
`level`) into `metadata` too, so a loaded node is a lossless copy of the source
row. Later phases consume these fields; the loader itself invents no per-node
supervision plumbing.

## `split_ids.json`

A JSON object mapping split name to a list of `doc_id`s:

```json
{ "train": ["<doc_id>", ...], "val": [...], "test": [...] }
```

When present, these ids are **authoritative**. Passing `split="train"` returns
exactly the trees whose `doc_id` is pinned to `train`, regardless of any
per-tree `metadata["split"]` value — splits are never resampled. When
`split_ids.json` is absent, the loader falls back to the per-tree
`metadata["split"]` field; if neither source exists, requesting a split raises.

## `manifest.json`

Optional top-level index. The loader reads it only to discover leaf-scale run
subdirectories. It typically records `target_dimensions`, `topology_axis`,
per-run `tree_counts`, and `split_ids` provenance. Its `corpus_csv` / `mpds_csv`
paths are producer-side and are not read by treepo.

## Versioning and validation

- **Required fields** (`doc_id`, `nodes` per tree; `node_id`, `level` per node)
  must be present. A missing one raises `BundleFormatError` naming the field and
  the file/line.
- **Unknown extra fields** at any level are tolerated and carried through, so a
  future `version` that adds fields still loads on today's code.
- The tree `version` is preserved as `metadata["schema_version"]` for callers
  that want to gate on it. `treepo.bundles.KNOWN_TREE_VERSIONS` lists the
  versions validated so far; values outside it are not rejected.

## What the loader produces

`load_labeled_tree_bundle` returns `list[TreeRecord]`. For each tree:

- `root_label` is the tree-level target `fit()` consumes — `document_score` by
  default, or the root node's `dimension_scores[dimension]` when a dimension is
  selected.
- Node `unit_type` is `"root"` for the deepest unparented node, `"leaf"` for
  level-0 (or `is_leaf`) nodes, and `"merge"` otherwise.
- Topology (`parent_id`, `left_child_id`, `right_child_id`, `level`, `position`)
  is reconstructed from child pointers and `levels`.

## Publishing to this contract

New tasks (Markov, Benoit-econ) become loadable by writing the same two files: a
`labeled_trees.jsonl` with the node schema above and a `split_ids.json` with
pinned partitions. No treepo change is needed to onboard a task that honors the
contract.
