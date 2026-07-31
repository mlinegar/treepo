# Trees and Sampling Over Leaves

Reference for how `treepo` builds trees, composes leaf states up to the
root, and samples nodes for supervision. It names the invariants the code
enforces and the module that owns each.

## The pipeline shape

A document `x` decomposes into ordered leaves. One operator `g` is invoked at
every state-producing call, and `f` reads the resulting root state:

```text
reduce_g(Leaf(b))       = g(b)
reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))
answer                   = f(reduce_g(T))
```

Thus `reduce_g` is a recursive program induced by `g`, not a separately learned
operator. Leaf and internal inputs may use typed tags or packing adapters, but
there is one `g` artifact, one parameter set, and one training surface. C1
leaf-call rows and C3 internal-call rows, when eligible, update that same
parameter set; there is no separate `train_reduce_g`. The same updated `g` is
used at both call types.

The generic provider-neutral `llm` family exposes fixed
text-state/concatenation behavior and does not optimize `g`. The
optimizer-backed `dspy` family learns one program over leaf and merge examples
and preserves this one-operator contract throughout recursive execution.

## Topology and the singleton base case

Every recursive binary C-Tree has `L >= 1` leaves and exactly `M = L - 1`
merge applications. With `L = 1`, the root and leaf are the same logical node
and `M = 0`. The package names three derived paths:

- `full_doc_direct`: singleton `f(reduce_g(T)) = f(X)` with one canonical
  identity call `g(X)=X`;
- `ctree_base_summary`: singleton `f(reduce_g(T)) = f(g(X))` with one nonidentity `g` call; and
- `ctree_recursive`: `f(reduce_g(T))` with `L >= 2` and active internal calls.

`full_doc` names full-span singleton geometry; `ctree` is the umbrella grammar
and therefore includes both singleton and recursive C-Trees.

In an artifact-aligned comparison, the recursive path is the only fitted
source. Every singleton view has `axis.max_iterations=0` and reuses its exact
`f`. The summarized singleton also reuses the exact source `g`; the direct
singleton substitutes only the package's canonical identity `g`. The
`model_artifact_contract` records source/scope IDs and validates these exact
choices package-wide.
In the summarized singleton, the sole `g(X)` call creates a realized C1
obligation. There are no internal nodes, so the realized C3 population is
empty. That is structural absence, not a C3 pass. An update changes the same
shared `g` that a larger tree would call internally, but singleton data supply
no internal-call or C3 evidence for learned composition.

More than one leaf with no merge reduction is not a recursive binary C-Tree.
For example, concatenating several leaf summaries directly into `f` is an
explicit concat/no-merge ablation and must be named separately.

The semantic local laws are C1 leaf validity, C2 recompression (only when the
grammar actually recompresses a stored state), and full state-level C3 merge
closure. `treepo.local_law` owns scalar row arithmetic and
`treepo.training.local_law` owns tensor objectives. Those rows may be faithful
state witnesses, separating probes, or scalar readout proxies; the arithmetic
and a C1/C2/C3 tag do not by themselves certify task sufficiency.

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

At `L = 1`, that formula gives one row: the root is the leaf. Sampling and
supervision code must not emit separate root and leaf node rows for that one
logical node. It may retain root-task and C1 evidence as distinct channels.
Three places implement this schedule and must stay in lockstep:

1. `_UnifiedGTreeModel._compose` / `_compose_batch` (`methods/_fno_models.py`) —
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

For node-level oracle audits, `treepo.sampling.sample_node_audit` draws a
uniform without-replacement design under a named policy (`all`, `fixed`,
`fraction`, `sqrt`, `log2`) and `apply_node_audit` realizes it on
fully-labeled audit rows: drawn rows keep their oracle losses, undrawn rows
keep only the proxy, and every row logs the design propensity `q / N`.

## The corrected objective

Training and audit can use the AIPW-corrected node loss

```text
loss_corrected = proxy + (observed / propensity) * (oracle − proxy)
```

which is unbiased for the declared oracle-loss estimand only when the logged
propensity is the correct conditional inclusion probability, the proxy is
fixed/predictable with respect to that selection, and denominator clipping is
inactive. It reduces to the proxy loss on unsampled nodes
(`treepo.local_law.corrected_local_law_loss`, tensor forms in
`treepo.training.local_law`). Aggregation weights each row by
`node_weight * gamma_depth ** depth` and normalizes by the total weight.
`node_weight` is the structural row weight; `gamma_depth` is the depth-emphasis
hyperparameter; neither changes logged propensities. Legacy certificate API
fields `delta` / `confidence_delta` store statistical tail allocation
(`alpha` in the current paper), not deployment relational risk `delta_T` and
not depth decay. `sampled_ipw` is the self-normalized Hájek ratio estimator
over observed rows; it is generally finite-sample biased and is not the
Horvitz--Thompson estimator. The
overall training objective is the convex combination
`(1 − lambda) * root_loss + lambda * corrected_law_loss`
(`treepo.objective.resolve_root_local_objective_weights`); audit diagnostics
(effective sample size, max IPW weight, propensity clipping) feed the
certificate, never the objective.

## Triangle/local-law error certificates

For partially observed trees, local-law rows provide a model-agnostic
*evidence input* to the legacy aggregate-distortion (`B_R`) lane. A point
objective estimates its declared realized row-population loss. It becomes a
leaf-up/root transport bound only after the caller supplies both a faithful or
calibrated row interpretation and a valid local-to-root transport argument;
using the same `f,g` calls at roots and internal nodes is not enough by itself.

Use `triangle_local_law_residual_from_audit(...)` to package either raw
`LocalLawAuditRow` values or an `audit_local_laws(...)` payload with a
caller-supplied bound in a `TwoChannelResidual`. Its channels map as follows:

- `leaf_up_radius`: a caller-supplied local-law/triangle bound;
- `root_down_radius`: root-label aggregate control through the rest-of-tree
  readout map;
- `overidentification_radius`: disagreement between leaf-up and root-down
  channels.

Use `build_triangle_local_law_error_certificate(...)` when the run also has
common-mechanism root-error envelopes or external conditional-average
envelopes. Pass a justified finite-sample/transport bound as `leaf_up_radius`.
The helper rejects an omitted bound by default. Descriptive legacy plots may
opt in to `allow_point_estimate_proxy_radius=True`; emitted metadata labels
that value a point-estimate proxy, not a semantic/root certificate.
The certificate preserves the audit's `gamma_depth` and effective-weight
formula in metadata so finite-sample bounds can be checked against the same
weighted estimand.

## Additive identification weights

`treepo.identification` defines the partially observed additive/share case.
For a node with mass `m` inside a document of mass `M`,
`additive_root_sensitivity(m, M) = m / M` and
`additive_root_information_weight(m, M) = (m / M)^2`. These are root-label
identification/sensitivity weights, not inclusion probabilities. They should
therefore be logged in metadata and, when a run opts in, passed as
`node_weight`; they must not overwrite `propensity` or any document/unit/label
sampling probability.

Use `annotate_additive_identification_rows(...)` after constructing local-law
rows to attach `node_mass`, `document_mass`, `additive_root_sensitivity`, and
`additive_root_information_weight`. The helper is model-agnostic: it accepts
ordinary `LocalLawAuditRow` values from any family, statistic, or audit
artifact. `profile="none"` preserves existing `node_weight`; `sensitivity`
uses `m/M` as `node_weight`, and `information` uses `(m/M)^2`. For
qsentence/CMP count or share labels, internal additive targets are derived
from leaf codes, so this is the exact additive special case rather than a
hand-observed internal-label regime.

## Visualizing trees

`treepo.viz.write_tree_visualization_html(trees, path, sampling_rows=...,
law_rows=...)` writes one self-contained HTML file with an expandable tree
per document: sampled/unsampled markers with propensities and IPW weights,
gold labels next to prediction metadata, text snippets and `g`-state
summaries, and local-law losses — statistic rows keyed by trace index render
on the synthesized merge tree. [`docs/visualization.md`](visualization.md)
documents the node display, input row shapes, and the Manifesto, Markov,
LDA, HLL, and generic reference views;
`examples/methods/run_tree_visualization.py` runs all five.

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
