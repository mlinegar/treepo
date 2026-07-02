# treepo

`treepo` is the Python package for C-TreePO: composable tree operators,
JSONable task states, unit-level supervision/preferences, local-law
certificates, lightweight LLM helpers, and small runnable benchmarks.

Large application code, model-serving fleets, datasets, launchers, and
publication-scale experiment grids live outside this package and register
through the public APIs.

## Quick Start

[uv](https://docs.astral.sh/uv/) is the supported project workflow:

```bash
git clone <this-repo-url> treepo
cd treepo
uv sync
uv run pytest -q
```

The default install includes the small numerical stack, Hugging Face
`datasets` for preference-data interchange, and the built-in neural-operator
family:

```bash
uv sync --no-dev
uv run python -c "import treepo; print(treepo.__version__)"
```

PyTorch and `neuraloperator` are installed by default for `family="neural_operator"` / `family="fno"`, but
`import treepo` does not import them.

Use `uv sync`, `uv run`, `uv lock`, and `uv build` so local checks match the
checked-in lockfile.

## Package Layout

```text
treepo/
├── src/treepo/
│   ├── bench/       # treepo-bench run/check implementation
│   ├── llm/         # OpenAI-compatible and Transformers helper contracts
│   ├── methods/     # internal fit registry and built-in lightweight families
│   ├── training/    # optional local-law tensor helpers
│   └── ...
├── examples/        # small bench and methods examples
├── docs/            # package boundary and architecture notes
├── tests/
└── inventory.yaml
```

## Install Extras

Core install stays import-light. Add extras for the workflow you are using:

```bash
uv sync --extra bench       # YAML config IO for treepo-bench
uv sync --extra sketches    # datasketches-backed sketch adapters
uv sync --extra llm         # OpenAI-compatible and Transformers clients
uv sync --extra train       # training/local-law helpers
uv sync --extra all         # every optional package stack
```

The `torch` and `neural` extras remain as explicit selectors for environments
that install extras piecemeal; the default package already includes them.

`import treepo` and `treepo.core` do not import PyYAML, langextract, tiktoken,
DSPy, OpenAI, vLLM, torch, pandas, transformers, or Hugging Face `datasets`.

## Fit Surface

`treepo.fit()` is the public learning entry point. It learns and evaluates one
composable tree-operator family from raw/tree traces plus optional
`PreferenceDataset` supervision.

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

Built-in families are deliberately small: deterministic oracles, a learnable
constant baseline, a concrete `fno` route, and a generic `neural_operator`
root scorer with `operator_kind="fno"`/`"fourier"`,
`operator_kind="tfno"`, `operator_kind="uno"`, and the local
`operator_kind="conv1d"` baseline. `llm` and `dspy` are provider-neutral
wrappers that require injected callables/programs. Additional application
families can register from the package or workspace that owns their runtime.
The package includes a small synthetic overlapping-topic LDA fixture with an
official sklearn baseline and leaf grouping-size grid as neural-operator
methods examples.

`PreferenceDataset` is the unit-level surface for root, node, merge,
trajectory, or task-unit candidate data. It stores one canonical Hugging Face
`DatasetDict` shape and exports generic, supervised, DPO, reward-model, and
GRPO-style downstream records.

`TaskState` is the JSONable value shape for explicit task states produced by
`g` and read by `f`. Exact sketches and learned operators may additionally
expose executable `ComposableStatistic` objects for encode/merge/readout and
local-law diagnostics.

## LLM Helpers

`treepo.llm` is a client-side helper layer for OpenAI-compatible chat/text
generation, embeddings, and native Transformers adapters. vLLM, SGLang,
OpenAI, and Transformers should feed the same normalized request/response
shape.

```python
lm_config = {
    "model": "openai/your-model",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
}
```

Server orchestration belongs outside `treepo`.

## Bench CLI

The public benchmark CLI has two commands:

```bash
treepo-bench run --help
treepo-bench check --help
```

Small runnable examples:

```bash
treepo-bench run classical-sketches \
  --config examples/bench/classical_sketches.yaml \
  --json-out outputs/classical_sketches.json \
  --csv-out outputs/classical_sketches.csv

treepo-bench run markov \
  --config examples/bench/markov.yaml \
  --json-out outputs/markov.json \
  --csv-out outputs/markov.csv
```

`markov` is a package-native task benchmark built on the built-in Markov
changepoint fixture and oracle. Synthetic overlapping-topic LDA is available as a small methods
example with an official sklearn baseline; manifesto workflows belong in downstream packages until they expose
the same native task benchmark contract.

Checks:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check release --json
```

## Examples

Runnable fixtures live under `examples/`:

| Example | Command |
| --- | --- |
| `examples/bench/classical_sketches.yaml` | `treepo-bench run classical-sketches` |
| `examples/bench/markov.yaml` | `treepo-bench run markov` |
| `examples/methods/*.toml` | `treepo.fit()` examples, including Markov and overlapping-topic synthetic LDA |

The Manifesto/RILE methods example has two lanes: root-only document labels for `f`, and document-unit `TaskState` labels for guiding `g`. The packaged fixture uses qsentences as document units; the same surface is intended for paragraph, section, or extractor-span units supplied by downstream tasks. Leaves group document units via `leaf_unit_count`.

## C-TreePO Shape

C-TreePO studies when a long document can be compressed through a tree without
losing task-relevant information:

```text
raw document x
  -> g: locally composable state sigma(x)
  -> f: downstream scorer/readout U(sigma(x))
```

The state must be locally mergeable. For example, a sketch state can be
merged locally and queried for a distinct-count estimate; a manifesto policy
state can preserve qsentence-level evidence and read out document-level RILE.

## Release Checks

Before treating this checkout as a release boundary, run:

```bash
uv lock --check
uv run pytest -q
uv run treepo-bench check release --json
uv run python -m treepo.release
uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts
```

The release check covers inventory, hygiene, public import laziness, example
config validation, and the benchmark CLI surface.

## References

- Package boundary: [`docs/boundary.md`](docs/boundary.md)
- Architecture: [`docs/architecture.md`](docs/architecture.md)
- `treepo.methods` module layout & decomposition convention: [`docs/methods_module_layout.md`](docs/methods_module_layout.md)
- LLM/code-agent guide: [`docs/llm_guide.md`](docs/llm_guide.md)
- Training defaults: [`docs/training_defaults.md`](docs/training_defaults.md)
