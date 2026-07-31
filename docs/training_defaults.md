# Fit Defaults And Extension Boundary

The package keeps the public learning layer deliberately small. It ships
`treepo.fit(...)`, lightweight defaults, deterministic oracle families, a
simple learnable family, a classical-sketch family, generic neural operators,
and provider-neutral LLM/DSPy wrappers. Heavier paper/application families plug
in from a downstream workspace or package.

## Built-In Surface

```python
import treepo

result = treepo.fit(
    {
        "family": "neural_operator",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "preference_data": preferences,
        "backend_config": {"operator_kind": "fno"},
        "axis": {"max_iterations": 3},
    }
)
```

The built-in families are:

| Family | Purpose |
|---|---|
| `oracle` | Wraps built-in oracle scorers as a `FamilyRuntime`. |
| `learnable_constant` | Tiny deterministic trainable baseline for package tests and API smoke. |
| `classical_sketch` | Exact classical sketch adapters (e.g. HLL) as a composable-statistic family. |
| `neural_operator` | Generic neural-operator root-score scorer over embedded leaf sequences; supports `operator_kind="fno"`, `operator_kind="tfno"`, `operator_kind="uno"`, and the local `operator_kind="conv1d"` baseline. |
| `fno` | Concrete FNO route over the shared neural-operator runtime. Use `family="neural_operator"` when selecting a non-FNO operator kind explicitly. |
| `llm` | Provider-neutral inference/artifact wrapper. Pass `api_base` for OpenAI-compatible servers such as vLLM/SGLang, or `predict_fn` for direct runtimes such as Transformers; it does not optimize `g`. |
| `dspy` | Provider-neutral optimizer-backed prompt-program family. Pass a non-disabled `optimizer` and `lm_config`; optional `f_program`/legacy `dspy_program` and `g_program` override the default joint-vector programs. |

The built-in oracles are:

| Oracle | Domain | Fixture |
|---|---|---|
| `hll_exact` | `classical_sketch` | `make_hll_item_trees(...)` |
| `markov_changepoint_count` | `markov` | `make_markov_changepoint_trees(...)` |

## Fit Pattern

Every public example uses the same call shape:

```python
treepo.fit(
    {
        "family": "learnable_constant",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "preference_data": preferences,
        "backend_config": {},
        "axis": {"max_iterations": 1, "axis_value": 0},
    },
)
```

Downstream code injects concrete callables/programs or registers real
families.

## Representation And `g` Contract

The family protocol exposes both `train_f(...)` and `train_g(...)`, but that
does not require every representation to train both sides.

A recursive binary C-Tree has `L >= 1` leaves and exactly `M = L - 1` merge
applications. `full_doc` names full-span singleton geometry; `ctree` is the
umbrella grammar. Identity `g` is defined only for the direct singleton path;
recursive composition requires an explicit fixed or learned state operator.

| Derived path | Tree shape | Merge applications | `g` status | Fit |
|---|---|---:|---|---|
| `full_doc_direct` | exactly one document-sized leaf | 0 | package-owned identity `g(X)=X`, one leaf call | execute `f(reduce_g(T))`; an aligned view reuses exact source `f` at zero iterations |
| `ctree_base_summary` | exactly one document-sized leaf | 0 | fixed or trainable nonidentity `g` | execute `f(reduce_g(T))=f(g(X))`; an aligned view reuses the exact source pair at zero iterations |
| `ctree_recursive` | `L >= 2` declared leaves | `L - 1` | the same fixed/analytic or trainable `g` at every call | execute `f(reduce_g(T))`; fit `f` and, when scheduled, the one shared `g`; composition evidence additionally requires internal-call/C3 support |

Experiment grids may use `ctree` as shorthand for their deliberately
multi-leaf arm, but that does not exclude the singleton C-Tree base case.
`reduce_g` is the fold induced by the same operator:
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`. It owns no
separate parameters, artifact, optimizer, or training schedule.
`treepo.align_model_artifacts(...)` and `treepo.fit(...,
artifact_source=source)` construct aligned evaluation views. They require zero
iterations and emit `model_artifact_contract` digests proving exact source
artifact reuse or the canonical identity substitution.


The public `g_mode` is the configured execution policy; the result's
`g_contract` is the execution outcome:

| `g_mode` | `schedule` | Result operator |
|---|---|---|
| `identity` | `f` | `fixed_identity`; direct singleton no-op state mechanism |
| `fixed` | `f` | `fixed_nonidentity_or_family_owned`; explicit deterministic/frozen control |
| `learned` | `fg` | `learned_shared` only after at least one `train_g` update |
| `learned` | `fg` | `learned_shared_reused` when verified initial artifacts are evaluated without an update |
| `learned` | `fg` | `trainable_not_updated_this_run` when the requested ladder never reaches a `g` update |

The runtime records `train_g_call_count` and `g_update_count` separately. A
family whose return artifact does not itself reveal an update should return
`treepo.methods.GTrainOutcome`; a no-op call never licenses
`learned_shared`. Fixed mode requires a concrete mapping artifact with a
non-empty `kind` and `operator`, `g_mode="fixed"`, and `trainable=false`; the
family validates that artifact before use. Identity artifacts are package
owned and cannot be injected or relabeled as fixed.

An omitted state mechanism in an `f`-only run means identity, not an unreported
learned operator; it says nothing about topology. A full-span singleton may
instead invoke a fixed or learned nonidentity `g` once and is then
`ctree_base_summary`, not the direct comparator. A deterministic mass-weighted
or other analytic rollup is the explicit `fixed` case and is scientifically
distinct from learned composition. Application-level chunking or merge calls
make the route `ctree_recursive`/segmented even if the source span is complete.

The generic provider-neutral `llm` family supplies direct scoring and a fixed
text-state/concatenation surface; it does not optimize `g`. The
optimizer-backed `dspy` family learns joint `f` and, on learned C-Tree paths,
one shared `g` throughout the fold. Optional `f_program`/legacy
`dspy_program` and `g_program` adapters override the default programs. The
result reports learning only after a realized update. At
`supervision_level="default"`, DSPy preserves its node-wide
`root_weight=leaf_weight=merge_weight=1`; named supervision levels use the
same `treepo.fit(...)` grid surface as the neural-operator families.

The paired Semantic-Forest grid uses `max_iterations=3` on its recursive
learned source cells, so they train `f -> g -> f`. Both singleton views execute
iteration zero: no optimizer slot runs, although their declared `g` still runs
during evaluation. Its learned DSPy backend sets
`f` update may therefore use supplied reference states before a learned `g`
exists, and the final `f` update consumes states generated by the current
shared `g`. `f_record_source="gold_state"` remains an explicit ablation that
keeps both `f` updates on reference/gold states; it is not the learned-grid
default.

This representation axis is orthogonal to target width. `K=1`, `K=3`, and
`K=57` use one ordered `oracle_targets` catalog, exact named-vector
prediction/target mappings, and the same point distance
`sum_j |prediction_j - target_j|`. `K=1` may mirror its sole coordinate into
legacy scalar report fields, but it does not use a different fit API or task
metric.

For `neural_operator` and `fno`, `backend_config["training_loss"]` selects
the coordinate reduction used by root targets, supervised node `f` readouts,
and numeric/task-supplied vector-state rows:

| Value | Status | Per-row definition |
|---|---|---|
| `"sum_l1"` | default | `sum_j abs(prediction_j - target_j)` |
| `"coordinate_mean_mse"` | explicit legacy compatibility | `mean_j (prediction_j - target_j)^2` |

Rows are subsequently averaged or combined under the declared root/node/law
weights; `sum_l1` never divides an individual row by `K`. Optional target
normalization still defines the coordinate space in which training runs, and
artifacts persist its center/scale as well as the loss name and definition.
This switch does not alter unrelated classification auxiliaries or the
canonical local-law/IPW estimator; it defines the vector loss rows fed to it.

## Supervision-Grid Axes

`treepo.fit` promotes supervision-grid knobs to first-class, validated spec
fields (top-level keys or `CTreePOLearningSpec` fields). Data-selection
defaults preserve today's behavior: all documents, no local-label mix, seed
0. Independently, `supervision_level="default"` applies no role-weight
override: neural-operator/FNO keeps its root-only 1/0/0 family default, while
DSPy keeps its node-wide 1/1/1 family default.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `doc_gold_n` | `int \| None` | `None` | How many documents contribute gold document-level labels. Drawn from one per-seed permutation and taken as a prefix, so cells at increasing `n` are nested (`25 ⊂ 50 ⊂ 100`) and the selected ids are pinned/persisted. `None` uses all documents. |
| `root_observed_doc_ids` | `sequence[str] \| None` | `None` | Opt-in independent root-observation mask. `None` preserves historical `doc_gold_n` training-pool subsetting. An explicit sequence (including empty) retains the complete training pool for local supervision and allows root loss only on those document ids. If `doc_gold_n` is also set, it must equal the sequence length. Currently supported by the neural-operator/FNO families. |
| `local_label_mix` | `"none" \| "gold_fraction" \| "llm_distilled"` | `"none"` | Node-level supervision. `none` = root-only. `gold_fraction` keeps gold node labels on a deterministic `p`-fraction of nodes (pinned per seed). `llm_distilled` routes to a cached-teacher node source. |
| `gold_fraction_p` | `float` | `1.0` | Kept-node fraction for `gold_fraction` (in `[0, 1]`). |
| `distilled_labels_path` | `str \| None` | `None` | Cached `teacher_node_rows.jsonl` source for `llm_distilled`. Absent one, supply a callable in `backend_config["node_oracle_predictor"]` (or `["predict_fn"]`); otherwise `fit()` errors naming what to configure. The cached-jsonl loader itself is Phase 2. |
| `seed` | `int` | `0` | One seed per `fit()` call; drives every pinned selection and seeds the backend when unset. |

```python
treepo.fit(
    {
        "family": "fno",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "doc_gold_n": 2,
        # Optional crossed-budget mode: keep every training tree while using
        # only these pinned document roots. Omit for historical subset mode.
        "root_observed_doc_ids": ["doc_004", "doc_019"],
        "local_label_mix": "gold_fraction",
        "gold_fraction_p": 0.5,
        "seed": 3,
    },
)
```

Each cell persists its axes into `summary["grid_axes"]`, the evidence JSON
(`evidence["grid_axes"]`, with the pinned `selected_doc_ids` and
`selected_node_units`), and the run manifest. To expand a full grid, use
`treepo.methods._grid_axes.expand_grid_cells(seeds=..., doc_gold_ns=...,
local_label_mixes=..., leaf_unit_counts=...)`, which emits one fully specified
cell per combination — `fit()` stays one seed per call.

When `root_observed_doc_ids` is explicit, `grid_axes.doc_gold` records
`observation_mode="masked_full_training_pool"`, the full training-pool count,
and the exact observed ids. Local-label selection is then resolved over that
full pool rather than over the root-labeled subset. Without the field,
`observation_mode="training_subset"` and all historical behavior is retained.

## Application Families

Downstream packages can register additional runtimes with
`treepo.methods.families.register_family(...)`. The package includes small
cardinality, Markov, overlapping-topic synthetic LDA, and Manifesto/RILE
fixtures plus generic family routes.

## Package Defaults

`treepo.methods.canonical_defaults` provides one generic helper used by source
examples and downstream packages:

| Name | Current role |
|---|---|
| `load_dataclass` | Hydrate any dataclass from TOML, with optional section selection and dotted-key overrides. Application families define their own default dataclasses in their own package and load them through this helper. |

## Release Rule

Adding a new model, scorer, oracle, or task should first be attempted as an
external registration against the existing contracts. Promote code into
`treepo` only when it is small, dependency-light, generally useful, and covered
by package tests.
