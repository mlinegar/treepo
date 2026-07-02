# Trees and Sampling Over Leaves

This is the system-level reference for how `treepo` builds trees, composes
leaf states up to the root, and samples nodes for supervision. It names the
invariants the code enforces and cites the module that owns each one.

## The pipeline shape

A document `x` decomposes into ordered leaves. `g` encodes each leaf into a
state and merges states pairwise up to a root state; `f` reads the root state
out into the task answer:

```text
raw document x
  -> leaves (b_1 .. b_L), in document order
  -> g: leaf states s_i = g(b_i); merges s = g(s_left, s_right)
  -> f: readout U(s_root)
```

The local laws (C1 leaf preservation, C2 on-range idempotence, C3 merge
preservation) certify that each `g` step preserves the task-relevant
information; `treepo.local_law` owns the scalar law arithmetic and audit rows,
`treepo.training.local_law` owns the tensor training objectives.

## Tree records: observation granularity

`treepo.tree.TreeRecord` stores what was observed: leaves, optional internal
nodes, labels/states, and JSONable metadata. Topology lives in two places and
both are honored:

- `left_child_id` / `right_child_id` slots, when a node has at most two
  children;
- `parent_id` edges, which carry topology for nodes of any width.

`TreeRecord.leaves()` treats a node as internal when it either points at
children or is named as another node's parent. `TreeRecord.root()` prefers an
explicit `unit_type="root"` node, then nodes that no edge points into.

Task fixtures (`treepo.tasks.manifesto`, `treepo.methods.fixtures`) produce
flat records — leaves plus a root — because the record captures observation
granularity. The merge topology over those leaves belongs to the family that
composes them (next section). A manifesto root with more than two leaves
lists every child in `metadata["child_node_ids"]` and leaves the binary
child-id slots empty.

Leaf grouping: `leaf_unit_count` groups adjacent document units (for example
qsentences) into one leaf, weight-averaging scores and concatenating text
(`tasks/manifesto/trees.py:_group_doc_units_into_leaves`,
`methods/fixtures/common.py:leaf_slices`). Leaf order is document order.

## Merge topology: one schedule, three implementations

The built-in neural-operator families compose leaf states with a single
schedule, defined in `methods/_fno_models.py` and mirrored exactly by the
supervision-target builder:

- adjacent states pair up as (0,1), (2,3), … in leaf order;
- an odd leftover state joins the next level after the merged states;
- the node trace lists every logical node exactly once: leaves first, then
  each merge level bottom-up, root last. A tree with `L` leaves yields
  `2L − 1` trace rows.

Three places implement this schedule and must stay in lockstep:

1. `_TreeFGModel._compose` / `_compose_batch` (`methods/_fno_models.py`) —
   the torch forward pass;
2. `_numeric_transition_rows` (`methods/_fno_transition.py`) — the exact
   per-node supervision targets;
3. `_pairwise_merge_depths` (`methods/_fno_transition.py`) — per-node depths
   in trace order.

`_numeric_transition_state_loss` and the statistic's `local_law_rows` raise
on any node-count mismatch between a model trace and its targets, so drift
between these implementations surfaces as an error. State width may differ:
the first target-width dimensions of the learned state carry the supervised
transition vector.

## Depth convention: root at depth 0

Node weights are `gamma_depth ** depth` with the root at depth 0, matching
the Lean formalization (`gamma = 0` collapses to root-only, `gamma = 1`
weights all nodes equally). Two conversions enforce this:

- `TreeRecord` levels count up from the leaves (leaf level 0, root highest),
  so `local_law_rows_from_tree_records` converts with
  `depth = max_level − level`;
- `_pairwise_merge_depths` reads each node's depth off its real parent edge,
  so a carried leftover node sits one level below the node that finally
  consumes it, deeper than a naive level count would say.

## Sampling over documents and leaves

`treepo.sampling` defines the propensity records; `tasks/manifesto/sampling.py`
shows the intended flow:

1. Sample documents uniformly without replacement
   (`sample_manifesto_replication_trees`). Every population member gets a
   `DocumentSamplingRow` with its inclusion probability; observed trees carry
   `document_propensity` in metadata.
2. Sample leaves within each observed document
   (`manifesto_document_unit_sampling_rows`), seeded per tree. The joint
   propensity is the product `document_propensity × unit_propensity ×
   label_propensity` (`SamplingMetadata.effective_joint_propensity`), and the
   IPW weight is its clipped reciprocal.

Propensities are logged design probabilities in `(0, 1]`; validation rejects
anything else. `MIN_PROPENSITY` (`treepo.common`) clips denominators only.

## The corrected objective

Training and audit both use the AIPW-corrected node loss

```text
loss_corrected = proxy + (observed / propensity) * (oracle − proxy)
```

which is unbiased for the oracle loss under logged propensities and reduces
to the proxy loss on unsampled nodes (`treepo.local_law.corrected_local_law_loss`,
tensor forms in `treepo.training.local_law`). Aggregation weights each row by
`node_weight * gamma_depth ** depth` and normalizes by the total weight.
`sampled_ipw` mode is the Hajek estimator over observed rows only. The
overall training objective is the convex combination
`(1 − lambda) * root_loss + lambda * corrected_law_loss`
(`treepo.objective.resolve_root_local_objective_weights`); audit diagnostics
(effective sample size, max IPW weight, propensity clipping) feed the
certificate, never the objective.

## Visualizing trees

`treepo.viz.write_tree_visualization_html(trees, path, sampling_rows=...,
law_rows=...)` writes one self-contained HTML file with an expandable tree
per document: sampled/unsampled markers with propensities and IPW weights,
gold labels next to prediction metadata, text snippets and `g`-state
summaries, and local-law losses — statistic rows keyed by trace index render
on the synthesized merge tree. [`docs/visualization.md`](visualization.md)
documents the node display, input row shapes, and the Manifesto, Markov, and
generic reference views; `examples/methods/run_tree_visualization.py` runs
all three.

## Performance notes

- Leaf encoding is cached per tree sequence
  (`NeuralOperatorFamily._encode_trees`): the cache is bounded, evicts oldest
  first, and pins the tree objects each entry was built from so identity keys
  stay valid. Text embedding dominates encoding cost; wrap the embedding
  client in `treepo.llm.DiskCachedEmbeddingClient` to reuse embeddings across
  runs and processes.
- The forward pass composes all equal-length trees in one batched
  level-by-level loop (`log2(L)` merge calls) and collects the node trace
  only when supervision or audit asks for it (`forward_with_trace`); plain
  `forward` skips trace assembly.
- Fixture generation supports `generation_device="cuda"` for large synthetic
  bundles (`methods/fixtures/markov.py`); generation is vectorized on device
  and materialized once.
