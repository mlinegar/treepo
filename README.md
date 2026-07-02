# treepo

`treepo` is the Python package for C-TreePO: composable tree operators,
JSONable task states, unit-level supervision/preferences, local-law
certificates, lightweight LLM helpers, and small runnable benchmarks.

Downstream packages register their runtimes, datasets, and experiment grids
through the public APIs.

## Quick Start

[uv](https://docs.astral.sh/uv/) is the supported project workflow:

```bash
git clone <this-repo-url> treepo   # replace <this-repo-url> with your checkout URL
cd treepo
uv sync
uv run pytest -q
```

The default install includes the numerical stack (`numpy`, `scipy`,
`scikit-learn`), Hugging Face `datasets` for preference-data interchange, and
the built-in neural-operator family (`torch`, `neuraloperator`):

```bash
uv sync --no-dev
uv run python -c "import treepo; print(treepo.__version__)"
```

PyTorch and `neuraloperator` install by default; they load only when a
neural-operator family runs.

Use `uv sync`, `uv run`, `uv lock`, and `uv build` so local checks match the
checked-in lockfile.

## Package Layout

```text
treepo/
├── src/treepo/
│   ├── bench/       # treepo-bench runner, benchmark IO, release checks
│   ├── llm/         # client-side OpenAI-compatible chat/embedding helpers
│   ├── methods/     # treepo.fit() registry and built-in families
│   ├── tasks/       # small task assets (Manifesto/RILE fixtures)
│   ├── training/    # torch local-law tensor helpers
│   └── *.py         # value modules: state, tree, statistic, local_law,
│                    #   evidence, certificate, objective, sampling,
│                    #   artifacts, finetune, common
├── examples/        # small bench and methods examples
├── docs/            # architecture, boundary, evidence, module layout,
│                    #   training defaults, and contributor guides
├── tests/
└── inventory.yaml
```

## Install Extras

Core install stays import-light. Add extras for the workflow you are using:

```bash
uv sync --extra bench       # YAML config IO for treepo-bench
uv sync --extra sketches    # datasketches-backed sketch adapters
uv sync --extra llm         # requests-based OpenAI-compatible client layer
uv sync --extra all         # every optional package stack
```

`import treepo` loads only the core modules; optional stacks load when the
workflow that uses them runs.

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

Seven families are built in, and each is deliberately small:

- `oracle` — deterministic built-in oracle scorers.
- `learnable_constant` — tiny trainable baseline for tests and API smoke.
- `classical_sketch` — exact classical sketch adapters (e.g. HLL).
- `neural_operator` — generic root scorer over embedded leaf sequences with
  `operator_kind="fno"`/`"fourier"`, `"tfno"`, `"uno"`, and the local
  `"conv1d"` baseline.
- `fno` — the concrete FNO route over the same runtime.
- `llm` and `dspy` — provider-neutral wrappers that accept injected
  callables/programs.

Additional application families register from the package or workspace that
owns their runtime. The package includes a small synthetic overlapping-topic
LDA fixture with an official sklearn baseline and a leaf-grouping grid as
neural-operator methods examples.

`PreferenceDataset` is the unit-level surface for root, node, merge,
trajectory, or task-unit candidate data. It stores one canonical Hugging Face
`DatasetDict` shape and exports generic, supervised, DPO, reward-model, and
GRPO-style downstream records.

`TaskState` is the JSONable value shape for explicit task states produced by
`g` and read by `f`. Exact sketches and learned operators may additionally
expose executable `ComposableStatistic` objects for encode/merge/readout and
local-law diagnostics.

## LLM Helpers

`treepo.llm` is a client-side helper layer built on `requests`: an
OpenAI-compatible chat payload helper plus embedding clients behind the
`EmbeddingClient` protocol —

- `OpenAICompatibleEmbeddingClient` — points at any OpenAI-compatible `/v1`
  endpoint (vLLM, SGLang, hosted providers).
- `HashingEmbeddingClient` — deterministic, dependency-free vectors for tests
  and smoke runs.
- `DiskCachedEmbeddingClient` — wraps another client with an on-disk cache so
  repeated sweeps embed each text once.
- `build_embedding_client(...)` — config-driven construction.

```python
lm_config = {
    "model": "openai/your-model",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
}
```

`treepo.llm` is client-side; the deploying package owns server orchestration.

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
changepoint fixture and oracle, and it shows the native task-benchmark
contract new tasks implement. Manifesto/RILE runs through the methods
examples, and synthetic overlapping-topic LDA is a small methods example with
an official sklearn baseline.

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
| `examples/methods/run_tree_visualization.py` | standalone expandable-tree HTML view of sampled trees |

The Manifesto/RILE methods example has two lanes: root-only document labels for `f`, and document-unit `TaskState` labels for guiding `g`. The packaged fixture uses qsentences as document units; the same surface is intended for paragraph, section, or extractor-span units supplied by downstream tasks. Leaves group document units via `leaf_unit_count`.

`treepo.viz.write_tree_visualization_html` renders tree records, sampling
rows, and local-law audit rows as one self-contained HTML file: an expandable
tree per document with sampled-leaf markers, propensities/IPW weights, gold
and prediction labels, text snippets, `g`-state summaries, and per-node
local-law losses on the computed merge tree. See
[`docs/visualization.md`](docs/visualization.md) for the Manifesto, Markov,
LDA, HLL, and generic reference views.

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
- Trees and sampling over leaves: [`docs/tree_and_sampling.md`](docs/tree_and_sampling.md)
- Tree visualization: [`docs/visualization.md`](docs/visualization.md)
- `treepo.methods` module layout & decomposition convention: [`docs/methods_module_layout.md`](docs/methods_module_layout.md)
- Anti-pattern catalog for cleanup passes: [`docs/antipatterns.md`](docs/antipatterns.md)
- Evidence artifact: [`docs/evidence.md`](docs/evidence.md)
- LLM/code-agent guide: [`docs/llm_guide.md`](docs/llm_guide.md)
- Training defaults: [`docs/training_defaults.md`](docs/training_defaults.md)
