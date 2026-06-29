# treepo

`treepo` is the Python package for C-TreePO: mergeable tree states,
local-law certificates, a compact methods surface, lightweight LLM helpers,
and small runnable benchmarks.

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

The default install includes the small numerical stack plus the built-in
neural-operator family:

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
│   ├── methods/     # fit/run registry and built-in lightweight families
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

`import treepo` and `treepo.core` do not require PyYAML, langextract, tiktoken,
DSPy, OpenAI, vLLM, torch, pandas, transformers, or datasets.

## Methods Surface

`treepo.methods` is the compact fit/run registry. The top-level `treepo.fit()`
is a thin learning entry point; benchmark examples run through
`treepo-bench run`.

```python
import treepo

treepo.list_methods()                      # ('audit', 'fit', 'oracle')
treepo.list_families()                     # ('fno', 'learnable_constant', 'neural_operator', 'oracle')
treepo.list_registered_oracles()           # ('hll_exact', 'markov_changepoint_count')

result = treepo.run("oracle", {"oracle_name": "hll_exact", "n_trees": 4})
```

Built-in families are deliberately small: deterministic oracles, a learnable
constant baseline, and a generic `neural_operator` root scorer with `operator_kind="fno"`, `operator_kind="tfno"`, `operator_kind="uno"`, and the local `operator_kind="conv1d"` baseline. `fno` is the short FNO alias. DSPy, TRL,
diffusion/dgemma and domain applications register from the package or
workspace that owns their application code. The package includes a small
synthetic overlapping-topic LDA fixture with an official sklearn baseline and leaf-size grid as neural-operator methods examples.

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
| `examples/methods/*.toml` | `treepo.fit()` / `treepo.methods.fit()` examples, including Markov and overlapping-topic synthetic LDA |


## C-TreePO Shape

C-TreePO studies when a long document can be compressed through a tree without
losing task-relevant information:

```text
raw document x
  -> locally composable state sigma(x)
  -> downstream scorer/readout U(sigma(x))
```

The state must be locally mergeable. For example, a sketch state can be
merged locally and queried for a distinct-count estimate.

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
- Training defaults: [`docs/training_defaults.md`](docs/training_defaults.md)
