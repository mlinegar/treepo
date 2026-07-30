# Per-cell `results.json` (results_version 0.1)

Every `fit()` run writes `results.json` to its `output_dir`, next to the run
manifest. It is the one artifact a grid cell contributes to cross-cell
comparison tooling (the ThinkingTrees W-ledger scripts). Writer:
`treepo.methods._results.write_results_json`.

For a Semantic Forest, `results.json` still describes one fit cell. Its
forest block records `definition="single_shared_g_joint_vector_f_star"`,
`target_order`, the ordered target/oracle records, the shared schema digest,
and oracle provenance. Native vector FNO `f`/`g` artifacts separately declare
their current training loss and target schema; width-one scalar-family
artifacts retain their native scalar representation behind the package's
vector boundary adapter. Predictions and metrics belong to one joint vector
readout and one declared `g`; there are no target child fits or
target-owned artifacts. `cell.g_mode` records configured policy;
`g_contract` records realized operator, fit status, warm-start presence, and
separate call/update counts. The mechanism is called learned in this run only
when `g_contract.learned_this_run=true`.

Topology provenance is independent of operator policy. Every binary C-Tree has
`L >= 1` leaves and `M = L - 1` merge applications. `full_doc` is full-span
singleton geometry, not a synonym for identity; `ctree` is the recursive
grammar and includes that singleton. `cell.execution_path` distinguishes:

- `full_doc_direct`: `f(X)` with identity `g` elided;
- `ctree_base_summary`: `f(g(X))`, with one `g` call and no internal call; and
- `ctree_recursive`: `f(reduce_g(T))` with `L >= 2` and `M >= 1`, where every
  leaf and internal node invokes the same `g` artifact.

`reduce_g` is derived recursion, not a second operator or learner:
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`.
Call-count fields may separate leaf and internal support, but all updates and
parameters belong to the one recorded `g` artifact.

`g_mode` records identity, fixed, or learned policy independently. A learned
operator records `operator=learned_shared` only after an executed update;
otherwise it is `operator=trainable_not_updated_this_run`. A singleton update
changes the shared `g` from leaf-call support only; it is not learned
composition. A deterministic
operator declares `g_mode=fixed` and a concrete fixed artifact. Neither an
unused `g` slot nor a skipped training stage may be reported as learned.
Representation, derived execution path, leaf count, merge-application count,
configured mode, and realized outcome remain visible when cells are compared.

The generic provider-neutral `llm` family does not optimize `g`; its fixed
text-state/concatenation artifact must not be reported as learned. The
optimizer-backed `dspy` family can compile one shared leaf/merge `g`, but a
learned claim still requires its explicit realized update and role-support
fields.

The learned Semantic-Forest grid defaults to `max_iterations=3` (`f -> g ->
f`). For DSPy it also records
`f_record_source="generated_when_available"`: the initial `f` may use
references before a learned `g` exists, and the final `f` consumes current-`g`
states. `f_record_source="gold_state"` denotes the explicit reference-state
ablation. These fields distinguish a final readout refit from a run that stops
immediately after updating `g`.

The oracle/evaluation point distance is the sum L1 distance at every width:
`sum_j |prediction_j-target_j|`. `K=1`, `K=3`, and `K=57` use the same API,
exact named-vector row contract, and reduction. At width one this is the same
absolute-error calculation, not a special scalar path. The dense
neural-operator/FNO family trains root, node-readout, and vector-state rows
with that same unnormalized sum-L1 reduction by default. Artifacts record the
loss name/definition; `coordinate_mean_mse` is an explicit legacy opt-in.
Per-dimension metrics remain named and unpooled diagnostics. Neither pooled
L1 alone is evidence that arbitrary cross-target dependencies were preserved.
A future dependence-sensitive joint loss or probe must receive
its own name/version and result block rather than being inferred from a pooled
coordinate metric.

For a one-target catalog, the results schema preserves the legacy scalar
readout and metric fields by unwrapping the sole coordinate. It also emits
named-vector mappings, target/oracle provenance, catalog digest, and per-target
mirror metrics. Legacy fields—including `teacher_vector`—remain unchanged; the
named target is carried separately in `target_by_name`. Thus `m = 1` is
computationally conservative after forgetting the additive schema, but its
`results.json` is not promised to be byte-identical to a legacy scalar file.
For `m > 1`, `prediction_scalar` is null and there is no implicit
first-coordinate, mean, or other scalar projection; any scalar summary must
name a separate estimand.

## Blocks

- **`cell`** — what was run: family, schedule, seed, axis, named supervision
  level + resolved weights, full grid-axes provenance (doc_gold selection,
  label mix + attach report), objective, final stage label. Representation
  reports additionally preserve leaf count and merge-application count.
  `g_mode` uses `identity|fixed|learned`; `g_contract.operator`,
  `fit_status`, `learned_this_run`, `initial_g_artifact_present`, and
  `train_g_call_count`/`g_update_count` describe attempts and realized updates
  separately.
- **`metrics`** — `pooled_across_dimensions: false` always;
  `splits.<split>` carries:
  - `external` / `internal`: `n`, `pearson_r`, `mae_native`, and
    `normalized_abs_error` = mean |ŷ−y| / (b−a) — the W1 `R_j` — with
    `scale_bounds` and `scale_bounds_source` recorded (`observed_gold_range`
    until a task supplies native bounds);
  - `joint`: the `l1` point metric, mean sum-absolute-coordinate distance,
    and the number of complete prediction/target vectors entering that
    mean;
  - `per_dimension`: unpooled per-dimension metrics (pooling across
    dimensions inflates Pearson and is banned);
  - `sim`: `theta_first_regime_accuracy`, `theta_last_regime_accuracy`,
    `contextual_mae` — the standing pairing, null when the cell has no sim
    channel but always present.
- **`local_laws`** — the evidence artifact's per-law summary
  (`summary`, `by_law_kind`, `source`).
- **`cost`** — three components, reported separately, never blended:
  - `label_cost`: gold doc labels consumed (pinned doc_gold count), node
    label source, gold vs distilled node-label counts, leaf/merge row counts;
  - `one_time_compute`: fit wall seconds, train tree count, iterations;
  - `marginal_inference`: eval prediction count;
  - `resummary_ops`: `{count, population}` — recorded even when zero
    (`empty_by_construction`), so an empty deployed-C2 stratum stays visible
    instead of reading as a pass. The same rule applies to every zero-merge
    singleton C-Tree, whether its state path is direct, fixed, or learned. A
    nonidentity singleton `g` call has a realized C1 population, but its C3
    population is structurally empty. A singleton `g` update therefore changes
    the shared operator from leaf-call support only; it is not a
    learned-composition result.
- **`paired_rows`** — pointers to the per-document
  `prediction_records/iter_*_post_eval.jsonl` files plus the field mapping
  (`key=tree_id`, `prediction=prediction_scalar`, `gold=expert_score`,
  `teacher=teacher_score`, `split=split`) that paired-Δ / bootstrap tooling
  ingests for legacy scalar fits. Named-vector fits additionally expose
  `named_vector_fields`, mapping the same row key/split to `target_order`,
  `prediction_by_target`, `target_by_name`, and `oracle_ids_by_target`. A
  singleton named vector also fills `prediction_scalar` with its sole
  coordinate so the legacy paired metrics remain available. With multiple
  targets, `prediction_scalar` is deliberately null and is never treated as
  coordinate zero or a coordinate average.
