# treepo

`treepo` is the Python package for the C-TreePO paper code: mergeable tree states, local-law certificates, HLL utilities, and a minimal methods surface. Benchmark and research code is intentionally namespaced under `treepo.bench` or `treepo._research` rather than exported from the package root. Intended as a clean v0.1.0 GitHub/source release artifact. PyPI packaging is not the launch target yet.

## Quick start

[uv](https://docs.astral.sh/uv/) is the canonical workflow. It manages the project `.venv/`, resolves dependencies from `uv.lock`, and is the only install/build path documented and release-tested for v0.1.0:

```bash
git clone <this-repo-url> treepo
cd treepo
uv sync                       # creates .venv, installs dev + research deps
uv run pytest tests/          # 287 passed on the v0.1.0 package tree
```

The slim runtime surface (just `import treepo`) requires only `numpy`. Non-minimal workflows pull in their own stacks: `bench` adds YAML config IO, `llm` adds OpenAI/DSPy plus `langextract`/`tiktoken`, and `train` adds the torch/transformers/TRL path. The default `uv sync` installs the dev dependency group, so the normal contributor environment has the LLM/research utilities available. For the minimal install:

```bash
uv sync --no-dev              # just core deps; dev / research tooling left out
```

Bare legacy installer commands are intentionally not part of the release workflow. Use `uv sync`, `uv run`, `uv lock`, and `uv build` so local checks match CI and the checked-in lockfile.

## Repo layout

```
treepo/
├── src/treepo/
│   ├── (core treepo modules: certificate, manifest, hll, ...)
│   ├── bench/                   # benchmark CLI implementation, suites, reports, runtime fixtures
│   ├── methods/                 # unified fit() / run() axis-factored API
│   └── _research/               # vendored research scaffolding (ctreepo/, core/,
│                                #   bench/, training/, tasks/, tree/, ... — was src/X)
├── scripts/                     # release-facing utility scripts
├── configs/research/methods/    # TOML fixtures for research/method examples
├── examples/research/           # bench, runtime, and method examples
├── docs/training_defaults.md
└── tests/
    ├── (treepo's own tests — 119 tests, release-gate hygiene clean)
    └── methods/                 # 154 tests for the unified surface
```

The `_research/` subpackage retains its `from treepo._research.X` import shape. It is listed in `release.MIGRATION_TIER_PREFIXES` so the hygiene gate doesn't flag its hardcoded local paths or heavy imports (those live with the research code, not the canonical package surface). Promoting individual modules out of `_research/` and into the canonical layout is the planned migration path.

## Backends: vLLM and SGLang

Both [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) are first-class local-chat backends. Treepo treats them symmetrically via the [`EngineRegistry`](src/treepo/_research/core/engines.py) (peer `EngineSpec` entries, both `launchable=True`, both `openai_compatible=True`).

### Configure paths

Three env vars resolve at import time via [`treepo.paths`](src/treepo/paths.py):

| Env var | Default | Purpose |
|---|---|---|
| `TREEPO_MODEL_DIR` | `~/models` | Root for local model snapshots (e.g. `$TREEPO_MODEL_DIR/google/embeddinggemma-300m`). |
| `TREEPO_VLLM_VENV` | `~/vllm-env` | venv where `vllm` is installed (launch scripts activate this). |
| `TREEPO_SGLANG_VENV` | `~/sglang-env` | venv where `sglang` is installed. |

Setting one of the venv vars flows end-to-end: `OrchestratorConfig.venv_path` reads it as a dataclass field default, so you don't have to thread it through every call site.

### Launching a local server

The [`scripts/`](scripts/) directory ships wrappers for both engines:

```bash
./scripts/start_vllm.sh                 # default model profile from config/settings.yaml
./scripts/start_vllm.sh qwen-80b        # explicit profile
./scripts/start_sglang.sh               # default sglang model profile
```

Both wrappers delegate to [`scripts/start_engine.py`](scripts/start_engine.py), which looks up the engine in `EngineRegistry`, sets `TT_START_ENGINE_DIRECT=1`, and exec's the chosen `launch_script`. To see the resolved engine spec without launching:

```bash
./scripts/start_engine.py --engine vllm   --print-spec
./scripts/start_engine.py --engine sglang --print-spec
```

The wrappers read `config/settings.yaml` for the model profile table (paths, tensor-parallel size, max context length). Treepo doesn't ship this file — copy from [`config/settings.example.yaml`](config/settings.example.yaml) and edit for your model layout.

### Connecting as a client

Most code in `treepo.methods` is a **client** for whichever engine is running. Point `DSPyFamilyConfig.lm_config` at the OpenAI-compatible base URL and it works the same regardless of backend:

```python
cfg = DSPyFamilyConfig(
    lm_config={"model": "openai/your-model",
               "api_base": "http://localhost:8000/v1",  # vLLM default
               # or "http://localhost:30000/v1",         # SGLang default
               "api_key": "EMPTY"},
)
```

### Parity guarantee

A parity test ([`tests/test_engine_parity.py`](tests/test_engine_parity.py)) loops over both engines and asserts:
- `launchable=True`, `openai_compatible=True`, `supports_profiles=True`
- `launch_script` file exists on disk
- `treepo.paths` exposes a venv-path helper for the engine

Adding a new local-chat backend means adding to that test's parametrize list.

## Unified `fit()` / `run()` interface (treepo.methods)

The axis-factored dispatcher and canonical-defaults framework lives at [`treepo.methods`](src/treepo/methods/) and is re-exported at the top level. The new surface:

```python
import treepo

treepo.list_methods()                          # ('audit', 'fit', 'oracle')
treepo.list_registered_oracles()               # all oracle names
treepo.list_oracle_domains_with_fixtures()     # ('classical_sketch', 'markov') — public auto-fixtures

# Hydrate any dataclass from TOML:
from treepo import load_dataclass
from treepo._research.ctreepo.fno_family import FNOFamilyConfig
cfg = load_dataclass("configs/research/methods/fno_smoke.toml", FNOFamilyConfig)

# Dispatch via the unified axis:
result = treepo.run("fit", {"family": "fno", "train_data": trees, "eval_data": trees,
                            "backend_config": {"fno_config": cfg, "output_dir": "out/"}, ...})
```

Per-family TOML configs live under [`configs/research/methods/`](configs/research/methods/), runnable examples under [`examples/research/methods/`](examples/research/methods/), and the canonical-defaults docs at [`docs/training_defaults.md`](docs/training_defaults.md).

## C-TreePO In One Paragraph

C-TreePO studies when a long document can be compressed through a tree without losing task-relevant information. The paper’s central shape is:

```text
raw document x
  -> locally composable state sigma(x)
  -> downstream scorer/readout U(sigma(x))
```

The state, not necessarily the final scalar score, must be locally mergeable. For example, an HLL register array is the mergeable state and the distinct-count estimate is the readout. A histogram is the mergeable state and a nonlinear utility can be the readout. C-TreePO's local laws certify that leaf encoders, merge operators, and re-summary operators preserve the task-relevant state fiber.

For the release-facing architecture overview see [`docs/architecture.md`](docs/architecture.md). The public paper link will be added once the manuscript posts.

## What Is In This Package

- `treepo.core`: experiment refs, role metadata, sampling plans, and canonical sidecar recording.
- `treepo.hll`: lightweight HLL utilities used by the core examples.
- `treepo.bench`: benchmark-only simulations, suite builders, result IO, reports, and LongBench runtime fixtures.
- `treepo.llm`: OpenAI-compatible payload/client helpers behind the `llm` extra.
- `treepo.training`: experiment methods exposing `train`, `evaluate`, and `predict`; PyTorch remains native and optional.
- `treepo.tasks`: minimal task-specific assets, currently including Manifesto/RILE constants.

The package includes lightweight local server wrapper scripts for vLLM and SGLang clients. It still excludes detached launchers, one-off overnight scripts, generated outputs, model checkpoints, and workspace-specific paths.

This is currently a source-tree release. The `examples/` and `docs/` directories are part of the GitHub checkout and are referenced by the launch instructions; they are not promised to be present from a wheel install until PyPI packaging is added.

## Install

The uv workflow is project-local; enter the package checkout first:

```bash
cd treepo
uv sync --no-dev
```

For the full development/research environment:

```bash
uv sync
```

Core install is intentionally small: `numpy`.

Optional extras:

```bash
uv sync --extra bench       # YAML config IO for treepo-bench/example configs
uv sync --extra reports     # matplotlib reports
uv sync --extra sketches    # datasketches-backed sketch adapters
uv sync --extra sklearn     # sklearn/proxy baselines
uv sync --extra torch       # learned sketch and state-model modules
uv sync --extra llm         # OpenAI/DSPy clients plus langextract/tiktoken
uv sync --extra runtime     # LongBench/RULER runtime helpers plus langextract
uv sync --extra research    # non-LLM research utilities, pandas/sklearn, langextract
uv sync --extra train       # DSPy/TRL/state-model training hooks
uv sync --extra all         # every optional package; add --no-dev to omit dev tools
uv sync --all-extras        # uv shorthand for all extras; default also includes dev
```

`import treepo` and `treepo.core` do not require PyYAML, langextract, tiktoken, DSPy, OpenAI, vLLM, torch, pandas, transformers, or datasets.

## Setup By Workflow

Use the smallest install that matches the workflow:

| Workflow | Install | Notes |
| --- | --- | --- |
| Import core/HLL APIs | `uv sync --no-dev` | Just `numpy`. |
| Run `paper-smoke` | `uv sync --extra bench --extra torch --extra sketches` | Exercises HLL, learned HLL merge, classical sketches, and mock LongBench runtime. |
| Run full simulation grids | `uv sync --extra bench --extra torch --extra sklearn --extra sketches` | Publication-scale grids can be large; emit commands first. |
| Generate reports/figures | `uv sync --extra reports` | Some report commands also need outputs from simulation grids. |
| Run live LLM scoring | `uv sync --extra llm --extra runtime` | LLM is the default non-minimal stack; requires an OpenAI-compatible chat endpoint for live scoring. |
| Develop/test package | `uv sync` | Full dev dependency group, including LLM/research utilities. |

For a clean package-only smoke from the parent workspace:

```bash
cd treepo
uv sync --extra bench --extra torch --extra sketches
uv run treepo-bench suite paper-smoke --out-root outputs/paper_smoke --jobs 1
uv run pytest -q
```

## Public API

```python
from treepo import (
    HLLConfig,
    HyperLogLogSketch,
    ExperimentContext,
    SamplingPlan,
    BenchmarkRef,
    MethodRef,
    role_ref,
    roles_metadata,
)
```

The experiment layer uses paper-facing roles:

- `scorer`: practical task scorer `f`
- `summarizer`: summarizer `g`
- `oracle`: trusted evaluator or benchmark labels `f*`
- `embedder`: vector evidence mechanism
- `state_model`: learned or deterministic state realization

Methods should expose `train`, `evaluate`, and `predict` at the experiment boundary. Raw PyTorch modules keep their native `module.train()` / `module.eval()` behavior inside a method wrapper; they are not treated as experiment methods directly.

Benchmark implementations are deliberately not re-exported from `treepo`.
Import benchmark classes/functions from `treepo.bench.*` or use the
`treepo-bench` CLI.

## CLI

The public CLI grammar is:

```bash
treepo-bench --help
treepo-bench run cardinality-recovery --config examples/research/bench/cardinality_recovery.yaml --json-out outputs/cardinality.json --csv-out outputs/cardinality.csv
treepo-bench suite cardinality-paper --out-root outputs/cardinality --jobs 4
treepo-bench report cardinality --output-root outputs/cardinality
```

Useful smoke commands:

```bash
treepo-bench suite cardinality-paper --out-root outputs/cardinality --jobs 1 --commands-only --seeds 0
treepo-bench suite paper-smoke --out-root outputs/paper_smoke --jobs 1
treepo-bench suite paper-grids --out-root outputs/paper_grids --jobs 1 --commands-only --emit-commands outputs/paper_grids/commands.sh
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check launch --json
```

## Centralized Paper Grids

The package has two central suite front doors.

`paper-smoke` is the quick executable check. It runs one tiny instance of each package-facing path:

- HLL/cardinality recovery.
- HLL merge learning.
- Classical sketch comparison.
- LongBench runtime with all package runtime methods.

```bash
treepo-bench suite paper-smoke \
  --out-root outputs/paper_smoke \
  --jobs 1
```

`paper-grids` is the publication-grid orchestrator. It composes the paper suite builders under one output root:

- `cardinality-paper`
- `classical-sketches`
- LongBench runtime smoke

For real publication-scale work, emit commands first and inspect the run count before launching:

```bash
treepo-bench suite paper-grids \
  --out-root outputs/paper_grids \
  --jobs 1 \
  --commands-only \
  --emit-commands outputs/paper_grids/commands.sh
```

Useful grid shrinkers:

```bash
treepo-bench suite paper-grids \
  --out-root outputs/paper_grids_debug \
  --jobs 1 \
  --commands-only \
  --emit-commands outputs/paper_grids_debug/commands.sh \
  --seeds 0 \
  --capacities small \
  --leaf-counts 1
```

Run generated commands with the scheduler of your choice. The package intentionally does not include local detached launchers or cluster queue management.

## Examples By Option

Small runnable fixtures live under `examples/research/` so they stay out of the minimal package surface:

| Option | Example | Status |
| --- | --- | --- |
| HLL / learned cardinality | `examples/research/bench/cardinality_recovery.yaml` | Runnable with `treepo-bench run cardinality-recovery` |
| HLL merge learning | `examples/research/bench/hll_merge_learning.yaml` | Runnable with `treepo-bench run hll-merge-learning` |
| Classical sketches | `examples/research/bench/classical_sketches.yaml` | Runnable with `treepo-bench run classical-sketches` |
| LLM full-context scoring | `examples/research/runtime/runtime_llm_full_context.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| Embedding retrieval + scorer | `examples/research/runtime/runtime_embedding_retrieval.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| Summary tree | `examples/research/runtime/runtime_summary_tree.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| FNO/state model + scorer | `examples/research/runtime/runtime_fno_state_model.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| All runtime methods | `examples/research/runtime/runtime_all_methods.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| LongBench v2 tiny row | `examples/research/runtime/longbench_v2_tiny.yaml` | Data fixture |

The runtime fixtures use the paper roles directly: `scorer`, optional `summarizer`, optional `embedder`, optional `state_model`, and `oracle`. They default to deterministic local/mock behavior for package smoke tests. Set `runtime_defaults.mock: false` and provide an OpenAI-compatible `scorer.endpoint`/`scorer.model` to use a live chat endpoint for final scoring.

## Current v0.1.0 Scope

Migrated now:

- HLL/cardinality recovery and HLL merge-learning experiments.
- Classical sketch comparison suite and reports.
- Lightweight experiment context, role metadata, LongBench fixture helpers, OpenAI-compatible payload helpers, and training lifecycle protocols.

Scaffolded but not yet the full workspace implementation:

- Full RULER runtime migration and live embedding/operator clients.
- Batched LongBench and Manifesto examples as first-class `treepo-bench suite` commands.
- LDA identifiable-zero suites now live in `treepo._research` rather than the public package surface.
- Neural/state-model training paths behind `treepo[torch]` and `treepo[train]`.

Known v0.1.0 certificate limitations:

- `treepo.certificate` emits a component-radius ledger for local-law, calibration, estimation, and clipping evidence.
- Lipschitz readout and measurement-error constants are not separate first-class components yet; include them in supplied radii when using theorem-complete bounds.
- HLL merge-learning diagnostics named `scalar_*` certify scalar readout behavior only, not state-level register-max Lean laws.

The migration inventory is checked in at `migration_inventory.yaml`; it classifies source candidates as `package_module`, `cli_command`, `compat_shim`, or `exclude_legacy`.

## Release Checks

Before treating this directory as the package release boundary, run:

```bash
uv run treepo-bench check inventory --json
uv run treepo-bench check hygiene --json
uv run treepo-bench check launch --json
uv run treepo-bench suite paper-smoke --out-root /tmp/treepo_paper_smoke --jobs 1
uv run treepo-bench suite paper-grids --out-root /tmp/treepo_paper_grids --jobs 1 --commands-only --emit-commands /tmp/treepo_paper_grids/commands.sh --seeds 0 --capacities small --leaf-counts 1
uv run pytest -q
uv build --wheel --out-dir /tmp/treepo_wheel_qc
```

The hygiene check rejects generated artifacts, package-local absolute paths, accidental imports from the parent workspace `src.*`, and heavyweight optional imports from the lightweight core surface.

The launch check aggregates inventory, hygiene, public import laziness, example-config validation, and paper-suite enumeration. The wheel command is a build sanity check only; source docs/examples remain the official launch surface for v0.1.0.

## Paper References

Workspace-local references:

- C-TreePO paper source: `../paper/ctreepo/main_new.tex`
- Minimal paper blueprint: `../paper/ctreepo/sections/minimal/BLUEPRINT.md`
- Preference scope statement: `../docs/preference_scope_ctreepo.md`
- Lean/paper map: `../lean3/docs/PAPER_TO_LEAN_MAP.md`
- Package architecture: `docs/architecture.md`

Public citation metadata will be added here once the paper has a stable public URL.
