# Tree Visualization

`treepo.viz.write_tree_visualization_html` renders trees as one
self-contained HTML file — an expandable tree per document, openable in any
browser with no server and no JavaScript dependencies. It reads the package's
existing artifacts, so any workflow that produces `TreeRecord`s, sampling
rows, or local-law audit rows can be inspected visually with one call.

```python
from treepo.viz import write_tree_visualization_html

write_tree_visualization_html(
    tree_records,                 # anything TreeRecord.from_value accepts
    "outputs/trees.html",
    sampling_rows=sampling_rows,  # optional: observed markers + propensities
    law_rows=law_rows,            # optional: per-node local-law losses
    readout_rows=readout_rows,    # optional: per-node f readouts
    audit=audit_payload,          # optional: audit_local_laws summary panel
    certificate=certificate_dict, # optional: error-certificate ledger panel
    title="my trees",
)
```

Run all five reference views with
`uv run python examples/methods/run_tree_visualization.py`; they land in
`outputs/tree_visualization/`.

## What each node shows

- **Collapsed row**: node id, a unit-type chip, a sampled (●) / unsampled (○)
  marker when sampling rows cover the node, a green `gold` chip for the
  node's label, blue chips for prediction-style metadata (`llm_score`,
  `prediction`, …; configurable via `label_keys=`), a purple `f→` chip for
  the node's readout, orange/red chips for local-law proxy/oracle losses,
  and an inline gray snippet of the node's text.
- **Expanded**: the full node text, green summary blocks for each summary the
  node carries, propensity/IPW facts, depth, and the full metadata and state
  JSON behind a `metadata` toggle.
- **Summaries** come from two places: the `text` field of a node's
  `TaskState` (the summary `g` wrote for that node) renders as a `state`
  block automatically, and string metadata under `summary`, `llm_summary`,
  `g_summary`, `state_text`, or `summary_text` (configurable via
  `summary_keys=`) render as labeled blocks.
- **Controls**: dim-unsampled toggle, expand all, collapse all. The tree
  header shows the root label, document sampling status and propensity, and
  node/leaf/sampled counts.

## Input row shapes

Sampling rows are mappings with a tree key (`tree_id` or `doc_id`), a node
key (`node_id`, or a `unit_id` in the `make_unit_id` spelling
`"<tree_id>:<node_id>"`), `observed`, and any of `document_propensity`,
`unit_propensity`, `joint_propensity`/`inclusion_probability`, `ipw_weight`,
`policy_name`. `manifesto_document_unit_sampling_rows` produces this shape.

Law rows are `LocalLawAuditRow` objects or their dicts, in either keying:

- **Node-keyed** — metadata carries `tree_id` and `node_id`
  (`local_law_rows_from_tree_records` produces this from `proxy_loss` /
  `oracle_loss` node metadata).
- **Trace-keyed** — `row_id` shaped `"<tree_id>:state:<node_index>"` (the
  neural-operator statistic's `local_law_rows` produces this).

Readout rows are mappings with a tree key, a `value`, and either a node key
or a `node_index` into the merge trace; the neural-operator statistic's
`node_readouts(trees)` produces the trace-indexed shape, with the last row
per tree equal to `predict_tree`. Rendering readouts at every node shows the
prediction forming up the tree toward the root gold label.

Rows attach only to the tree they declare. Rows that declare no tree apply
when a single tree is rendered; in a multi-tree file they are dropped as
ambiguous.

## Audit and certificate panels

`audit=` takes the payload `audit_local_laws(rows, ...)` returns and renders
a summary table above the trees: the AIPW-corrected (or sampled-IPW)
objective, row and observed counts, Kish effective sample size, and max IPW
weight, overall and per law kind. `certificate=` takes an error-certificate
dict (`UnifiedLearningErrorCertificate.to_dict()`) and renders the
component-radius ledger with the total bound.

To audit a subset of nodes under a logged design, draw the design with
`treepo.sampling.sample_node_audit` (policies `all`, `fixed`, `fraction`,
`sqrt`, `log2`; every node logs inclusion probability `q/N`) and realize it
with `apply_node_audit(rows, design)`: drawn rows keep their oracle losses,
undrawn rows keep only the proxy, and all rows log the design propensity —
exactly the shape the corrected objective and the audit panel expect.

## Trace rows synthesize the merge tree

Task fixtures store trees as flat stars — leaves plus a root — because the
record captures observation granularity while the merge topology belongs to
the family that composes the leaves. Trace-keyed law rows index nodes of
that computed topology: leaves `0..L-1` in position order, then each merge
level bottom-up, root state last (index `2L−2`).

When a flat record carries trace-keyed rows, the view synthesizes the
intermediate `merge_<index>` nodes from the shared pairwise schedule
(`_pairwise_merge_children` in `treepo.methods._fno_transition`, the same
definition the depth helper uses) and attaches each loss to its real node —
so you see proxy losses on the tree the model actually computed, root at
depth 0. If the trace indices and the record's leaf count disagree, the call
raises rather than attaching losses to the wrong nodes.

## The five reference views

**Manifesto** (`write_manifesto_view`) — sampling design over real text:

```python
trees = make_manifesto_replication_trees(split="test", leaf_unit_count=1)
observed, _ = sample_manifesto_replication_trees(trees, sample_rate=0.75, seed=0)
unit_rows = manifesto_document_unit_sampling_rows(observed, sample_rate=0.5, seed=0)
write_tree_visualization_html(manifesto_tree_records(observed), path,
                              sampling_rows=unit_rows)
```

Shows which qsentences the design sampled (with propensities and IPW
weights), gold qsentence and root RILE labels, and each leaf's policy-state
summary.

**Markov** (`write_markov_law_view`) — the full loop: learned family, node
audit design, AIPW summary, and per-node readouts against exact supervision:

```python
trees = make_markov_changepoint_trees(...)
family = resolve_family("neural_operator", {..., "numeric_transition_state_weight": 0.05})
# train f then g, then per tree:
rows = statistic.local_law_rows([tree])
design = sample_node_audit(len(rows), policy="sqrt", seed=tree_idx)
audited.extend(apply_node_audit(rows, design))
# and once:
write_tree_visualization_html(
    markov_tree_records(trees), path,
    law_rows=audited,
    readout_rows=statistic.node_readouts(trees),
    audit=audit_local_laws(audited, gamma_depth=0.9),
)
```

`markov_tree_records` converts fixture trees into records whose leaves carry
exact within-leaf changepoint counts as gold labels, so the view puts the
model's readouts and audited local-law losses next to the ground truth on
the synthesized merge tree, with the AIPW summary as a panel.

**LDA** (`write_lda_readout_view`) — `lda_tree_records` labels each leaf
with its realized target-topic proportion (full per-leaf topic mix in
metadata) and the root with the exact document proportion; the view shows
the trained family's readouts converging toward it.

**HLL** (`write_hll_view`) — `hll_tree_records` labels each leaf with its
exact within-leaf distinct count and the root with the exact document
distinct count: the mergeable-sketch ground truth at every node.

**General** (`write_generic_view`) — bring your own records. Any
`TreeRecord` works: put gold values in node `label`, predictions and
summaries in node metadata, `proxy_loss`/`oracle_loss` in node metadata to
generate node-keyed law rows via `local_law_rows_from_tree_records`, and
sampling rows in the shape above. This is the path for downstream tasks
(paragraphs, sections, extractor spans) and for eyeballing LLM labels
against gold on any corpus.
