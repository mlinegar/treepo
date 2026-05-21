# treepo

`treepo` is the Python package for the C-TreePO paper code: mergeable tree states, sketch baselines, simulation suites, benchmark reports, runtime scaffolds, and training hooks. Intended as a clean v0.1.0 GitHub/source release artifact. PyPI packaging is not the launch target yet.

## Quick start

```bash
git clone <this-repo-url> treepo
cd treepo
pip install -e .              # core (numpy + pyyaml)
pip install -e ".[all]"       # everything including sklearn, torch, dspy, ...
pytest tests/                 # 273 tests, ~30s on a laptop
```

## Repo layout

```
treepo/
├── src/treepo/
│   ├── (core treepo modules: audit, certificate, manifest, hll, ...)
│   ├── cld/                 # unified fit() / run() axis-factored API
│   └── _research/           # vendored research scaffolding (ctreepo/, core/,
│                            #   training/, tasks/, tree/, ... — was src/X)
├── scripts/                 # probe_clean_unified_no.py, run_lda_tree_recovery_simulation.py
├── configs/cld/             # 7 TOML configs (one per family)
├── examples/cld/            # 7 runnable examples
├── docs/training_defaults.md
└── tests/
    ├── (treepo's own tests — 119 tests, release-gate hygiene clean)
    └── cld/                 # 154 tests for the unified surface
```

The `_research/` subpackage retains its `from treepo._research.X` import shape. It is listed in `release.MIGRATION_TIER_PREFIXES` so the hygiene gate doesn't flag its hardcoded local paths or heavy imports (those live with the research code, not the canonical package surface). Promoting individual modules out of `_research/` and into the canonical layout is the planned migration path.

## Unified `fit()` / `run()` interface (treepo.cld)

The axis-factored dispatcher and canonical-defaults framework lives at [`treepo.cld`](src/treepo/cld/) and is re-exported at the top level. The new surface:

```python
import treepo

treepo.list_methods()                          # ('audit', 'fit', 'oracle', 'probe', 'sketch')
treepo.list_registered_oracles()               # all oracle names
treepo.list_oracle_domains_with_fixtures()     # ('classical_sketch', 'lda', 'markov') — all auto-build

# Hydrate any dataclass from TOML:
from treepo import load_dataclass
from treepo._research.ctreepo.fno_family import FNOFamilyConfig
cfg = load_dataclass("configs/cld/fno_smoke.toml", FNOFamilyConfig)

# Dispatch via the unified axis:
result = treepo.run("fit", {"family": "fno", "train_data": trees, "eval_data": trees,
                            "backend_config": {"fno_config": cfg, "output_dir": "out/"}, ...})
```

Per-family TOML configs live under [`configs/cld/`](configs/cld/), runnable examples under [`examples/cld/`](examples/cld/), and the canonical-defaults docs at [`docs/training_defaults.md`](docs/training_defaults.md).

## C-TreePO In One Paragraph

C-TreePO studies when a long document can be compressed through a tree without losing task-relevant information. The paper’s central shape is:

```text
raw document x
  -> locally composable state sigma(x)
  -> downstream scorer/readout U(sigma(x))
```

The state, not necessarily the final scalar score, must be locally mergeable. For example, an HLL register array is the mergeable state and the distinct-count estimate is the readout. A histogram is the mergeable state and an LDA likelihood or nonlinear utility can be the readout. C-TreePO’s local laws certify that leaf encoders, merge operators, and re-summary operators preserve the task-relevant state fiber.

For the release-facing architecture overview see [`docs/architecture.md`](docs/architecture.md). The public paper link will be added once the manuscript posts.

## What Is In This Package

- `treepo.core`: experiment refs, role metadata, sampling plans, and canonical sidecar recording.
- `treepo.hll` and `treepo.sketches`: HLL and classical sketch protocols/adapters.
- `treepo.bench`: paper simulations, suite builders, result IO, and report generators.
- `treepo.runtime`: LongBench/RULER-style runtime scaffolds and fixture helpers.
- `treepo.llm`: OpenAI-compatible payload/client helpers behind `treepo[llm]`.
- `treepo.training`: experiment methods exposing `train`, `evaluate`, and `predict`; PyTorch remains native and optional.
- `treepo.tasks`: minimal task-specific assets, currently including Manifesto/RILE constants.

The package deliberately excludes local server management, detached launchers, one-off overnight scripts, generated outputs, model checkpoints, and workspace-specific paths.

This is currently a source-tree release. The `examples/` and `docs/` directories are part of the GitHub checkout and are referenced by the launch instructions; they are not promised to be present from a wheel install until PyPI packaging is added.

## Install

From the parent workspace:

```bash
pip install -e ./treepo
```

From inside this package directory:

```bash
pip install -e .
```

Core install is intentionally small: `numpy` and `pyyaml`.

Optional extras:

```bash
pip install -e "./treepo[reports]"    # matplotlib reports
pip install -e "./treepo[sklearn]"    # sklearn LDA/proxy baselines
pip install -e "./treepo[torch]"      # learned sketch and state-model modules
pip install -e "./treepo[llm]"        # OpenAI-compatible clients and batching helpers
pip install -e "./treepo[runtime]"    # LongBench/RULER runtime helpers
pip install -e "./treepo[train]"      # DSPy/TRL/state-model training hooks
pip install -e "./treepo[all]"        # everything, including dev tools
```

`import treepo`, `treepo.core`, and `treepo.sketches` do not require DSPy, OpenAI, vLLM, torch, pandas, transformers, or datasets.

## Setup By Workflow

Use the smallest install that matches the workflow:

| Workflow | Install | Notes |
| --- | --- | --- |
| Import core/HLL/sketch APIs | `pip install -e ./treepo` | No torch, sklearn, matplotlib, or LLM packages. |
| Run `paper-smoke` | `pip install -e "./treepo[torch,sketches]"` | Exercises HLL, learned HLL merge, classical sketches, LDA embedding-spectral, and mock LongBench runtime. |
| Run full simulation grids | `pip install -e "./treepo[torch,sklearn,sketches]"` | Publication-scale grids can be large; emit commands first. |
| Generate reports/figures | `pip install -e "./treepo[reports]"` | Some report commands also need outputs from simulation grids. |
| Run live LLM scoring | `pip install -e "./treepo[llm,runtime]"` | Requires an OpenAI-compatible chat endpoint. Mock runtime examples do not. |
| Develop/test package | `pip install -e "./treepo[all]"` | Full optional stack plus dev tooling. |

For a clean package-only smoke from the parent workspace:

```bash
python3 -m venv .venv-treepo
source .venv-treepo/bin/activate
pip install -U pip
pip install -e "./treepo[torch,sketches,dev]"
cd treepo
PYTHONPATH=src treepo-bench suite paper-smoke --out-root outputs/paper_smoke --jobs 1
PYTHONPATH=src python -m pytest -q
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

## CLI

The public CLI grammar is:

```bash
treepo-bench --help
treepo-bench run cardinality-recovery --config examples/cardinality_recovery.yaml --json-out outputs/cardinality.json --csv-out outputs/cardinality.csv
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
- LDA embedding-spectral C-TreePO.
- LongBench runtime with all package runtime methods.

```bash
treepo-bench suite paper-smoke \
  --out-root outputs/paper_smoke \
  --jobs 1
```

`paper-grids` is the publication-grid orchestrator. It composes the paper suite builders under one output root:

- `cardinality-paper`
- `classical-sketches`
- `identifiable-zero-dtm-lda`
- `identifiable-zero-lda-leafnoise`
- `identifiable-zero-publication-ctreepo`
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
  --leaf-counts 1 \
  --topic-phi-estimators tensor_lda
```

Run generated commands with the scheduler of your choice. The package intentionally does not include local detached launchers or cluster queue management.

## Examples By Option

Small examples live in `examples/`:

| Option | Example | Status |
| --- | --- | --- |
| HLL / learned cardinality | `examples/cardinality_recovery.yaml` | Runnable with `treepo-bench run cardinality-recovery` |
| HLL merge learning | `examples/hll_merge_learning.yaml` | Runnable with `treepo-bench run hll-merge-learning` |
| Classical sketches | `examples/classical_sketches.yaml` | Runnable with `treepo-bench run classical-sketches` |
| Local embedding-spectral LDA | `examples/lda_embedding_spectral.yaml` | Runnable with `treepo-bench run segmented-lda-ctreepo` |
| LLM full-context scoring | `examples/runtime_llm_full_context.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| Embedding retrieval + scorer | `examples/runtime_embedding_retrieval.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| Summary tree | `examples/runtime_summary_tree.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| FNO/state model + scorer | `examples/runtime_fno_state_model.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| All runtime methods | `examples/runtime_all_methods.yaml` | Runnable with `treepo-bench run longbench-runtime` |
| LongBench v2 tiny row | `examples/longbench_v2_tiny.yaml` | Data fixture |

The runtime fixtures use the paper roles directly: `scorer`, optional `summarizer`, optional `embedder`, optional `state_model`, and `oracle`. They default to deterministic local/mock behavior for package smoke tests. Set `runtime_defaults.mock: false` and provide an OpenAI-compatible `scorer.endpoint`/`scorer.model` to use a live chat endpoint for final scoring.

## Current v0.1.0 Scope

Migrated now:

- HLL/cardinality recovery and HLL merge-learning experiments.
- Classical sketch comparison suite and reports.
- Existing LDA identifiable-zero suites, pending further file-level splitting.
- Lightweight experiment context, role metadata, LongBench fixture helpers, OpenAI-compatible payload helpers, and training lifecycle protocols.

Scaffolded but not yet the full workspace implementation:

- Full RULER runtime migration and live embedding/operator clients.
- Batched LongBench and Manifesto examples as first-class `treepo-bench suite` commands.
- Neural/state-model training paths behind `treepo[torch]` and `treepo[train]`.

The migration inventory is checked in at `migration_inventory.yaml`; it classifies source candidates as `package_module`, `cli_command`, `compat_shim`, or `exclude_legacy`.

## Release Checks

Before treating this directory as the package release boundary, run:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check launch --json
treepo-bench suite paper-smoke --out-root /tmp/treepo_paper_smoke --jobs 1
treepo-bench suite paper-grids --out-root /tmp/treepo_paper_grids --jobs 1 --commands-only --emit-commands /tmp/treepo_paper_grids/commands.sh --seeds 0 --capacities small --leaf-counts 1 --topic-phi-estimators tensor_lda
python -m pytest -q
python -m pip wheel --no-deps -w /tmp/treepo_wheel_qc .
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
