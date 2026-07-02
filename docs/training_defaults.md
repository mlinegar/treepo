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
| `llm` | Provider-neutral prompt wrapper. Pass `predict_fn` for concrete OpenAI-compatible/vLLM calls. |
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
