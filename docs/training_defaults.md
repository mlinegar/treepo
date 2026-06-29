# `treepo.methods` Defaults And Extension Boundary

The v0.1 package keeps the public methods layer deliberately small. It ships
the dispatch contract, lightweight defaults, deterministic oracle families, a
simple learnable family, generic neural operators, and provider-neutral
LLM/DSPy wrappers. Heavier paper/application families plug in from a downstream
workspace or package.

## Built-In Surface

```python
import treepo

treepo.list_methods()                      # ("audit", "fit", "oracle")
treepo.list_families()                     # includes "dspy", "fno", "llm", "neural_operator", ...
treepo.list_registered_oracles()           # ("hll_exact", "markov_changepoint_count")
treepo.list_oracle_domains_with_fixtures() # ("classical_sketch", "markov")
```

The built-in families are:

| Family | Purpose |
|---|---|
| `oracle` | Wraps built-in oracle scorers as a `FamilyRuntime`. |
| `learnable_constant` | Tiny deterministic trainable baseline for package tests and API smoke. |
| `neural_operator` | Generic neural-operator root-score scorer over embedded leaf sequences; supports `operator_kind="fno"`, `operator_kind="tfno"`, `operator_kind="uno"`, and the local `operator_kind="conv1d"` baseline. |
| `fno` | Short route for `neural_operator` with `operator_kind="fno"`. |
| `llm` / `prompted_llm` | Provider-neutral prompt wrapper. Pass `predict_fn` for concrete OpenAI-compatible/vLLM calls. |
| `dspy` | Provider-neutral DSPy wrapper. Pass `dspy_program`, `program`, or `predict_fn`; importing it does not import DSPy. |

The built-in oracle is:

| Oracle | Domain | Fixture |
|---|---|---|
| `hll_exact` | `classical_sketch` | `make_hll_token_trees(...)` |
| `markov_changepoint_count` | `markov` | `make_markov_changepoint_trees(...)` |

## Dispatch Pattern

Every public method is one axis, not a new method per experiment:

```python
treepo.run("oracle", {"oracle_name": "hll_exact", "n_trees": 4})

treepo.run(
    "fit",
    {
        "family": "learnable_constant",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "backend_config": {},
        "axis": {"max_iterations": 1, "axis_value": 0},
    },
)
```

Downstream code should inject concrete callables/programs or register real
families instead of adding package-level branches.

## Optional Application Families

These names remain optional application families. The package recognizes them
and raises a clear `ImportError` until a downstream package registers them:

- `trl`
- `diffusion`
- `dgemma`
- `diffusiongemma`

Specialized LDA recovery, large manifesto training campaigns, neural-operator
state models, and diffusion/generate experiments are extension packages. The
package includes small cardinality, Markov, overlapping-topic synthetic LDA, and
Manifesto/RILE fixtures plus generic estimator routes.

## Package Defaults

`treepo.methods.canonical_defaults` provides small constants and TOML
helpers used by source examples and downstream packages:

| Name | Current role |
|---|---|
| `load_dataclass` | Hydrate a dataclass from TOML without adding a new config layer. |
| `LmSection` / `build_lm_config_dict` | Lightweight OpenAI-compatible client config helper. |
| `BATCH_DEFAULTS` | Conservative batch-client defaults for downstream LLM families. |
| `GEPA_STRONG_DEFAULTS` | Small GEPA defaults for downstream DSPy/GEPA code. |

Those constants are literal package defaults.

## Release Rule

Adding a new model, scorer, oracle, or task should first be attempted as an
external registration against the existing contracts. Promote code into
`treepo` only when it is small, dependency-light, generally useful, and covered
by package tests.
