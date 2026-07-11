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
        "axis": {"max_iterations": 2},
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
| `llm` | Provider-neutral prompt wrapper. Pass `api_base` for OpenAI-compatible servers such as vLLM/SGLang, or `predict_fn` for direct runtimes such as Transformers. |
| `dspy` | Provider-neutral DSPy wrapper. Pass `dspy_program`, `program`, or `predict_fn`; DSPy loads only when a program runs. |

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

## Supervision-Grid Axes

`treepo.fit` promotes three supervision-grid knobs to first-class, validated
spec fields (top-level keys or `CTreePOLearningSpec` fields). Defaults preserve
today's behavior: all documents, root-only labels, seed 0.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `doc_gold_n` | `int \| None` | `None` | How many documents contribute gold document-level labels. Drawn from one per-seed permutation and taken as a prefix, so cells at increasing `n` are nested (`25 ⊂ 50 ⊂ 100`) and the selected ids are pinned/persisted. `None` uses all documents. |
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
        "doc_gold_n": 25,
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
