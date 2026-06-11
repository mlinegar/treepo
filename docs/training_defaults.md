# `treepo.methods` paper-canonical defaults

One pattern across every family. No family has special logic the others don't.

## The pattern (three lines, every family)

```python
from treepo.methods.canonical_defaults import load_dataclass
from <upstream_or_canonical_module> import <ConfigDataclass>

cfg = load_dataclass("configs/research/methods/<family>.toml", <ConfigDataclass>)
treepo.methods.run("<method>", {"backend_config": {..., "<thing>_config": cfg}, ...})
```

If a family has an **upstream dataclass** (FNO, DSPy, Markov DGP),
the TOML loads directly into it -- no mirror, no translator.
If a family doesn't have an upstream dataclass, keep that config local to
the research or bench surface rather than adding another public method axis.

## Families covered

| Family | Method | Loaded type | TOML | Example |
|---|---|---|---|---|
| **DSPy / LLM** | `run("fit", {family="dspy", ...})` | `DSPyFamilyConfig` *(upstream)* + `LmSection` | [`configs/research/methods/manifesto_fg_compile.toml`](../configs/research/methods/manifesto_fg_compile.toml) | [`examples/research/methods/run_manifesto_fg_compile.py`](../examples/research/methods/run_manifesto_fg_compile.py) |
| **FNO family** | `run("fit", {family="fno", ...})` | `FNOFamilyConfig` *(upstream)* | [`configs/research/methods/fno_smoke.toml`](../configs/research/methods/fno_smoke.toml) | [`examples/research/methods/run_fno_family.py`](../examples/research/methods/run_fno_family.py) |
| **Markov change-point oracle** | `run("oracle", {oracle_name="markov_changepoint_count", ...})` | `MarkovChangepointConfig` *(upstream)* | [`configs/research/methods/markov_oracle.toml`](../configs/research/methods/markov_oracle.toml) | [`examples/research/methods/run_markov_oracle.py`](../examples/research/methods/run_markov_oracle.py) |

*Upstream* = the dataclass already exists in the vendored research tree
(`src/treepo/_research/...`).
The TOML loads into it directly. No mirror code in `treepo.methods`.

## Adding a new family

Cost: one TOML + a ~30-line example script. Public methods should use an
upstream dataclass where possible; ad-hoc benchmark configs belong under
`configs/research/` or `treepo.bench`.

That's the entire surface. No `RunConfig` wrappers, no
`build_*_config_dict` translators, no per-family drift tests. The
parametrized drift test grows by one line per family.

---

## Sources of truth

| Family | Upstream truth (loaded directly) |
|---|---|
| DSPy / LLM family | `src/treepo/_research/ctreepo/dspy_family.py::DSPyFamilyConfig` |
| FNO family | `src/treepo/_research/ctreepo/fno_family.py::FNOFamilyConfig` |
| Markov change-point DGP | `src/treepo/_research/tree/markov_changepoint_honesty_simulation.py::MarkovChangepointConfig` |
Cross-family constants (re-exported in `canonical_defaults.py` and pinned
by drift tests):

| Constant | Upstream |
|---|---|
| `GEPA_STRONG_DEFAULTS` | `src/treepo/_research/training/gepa_defaults.py::GEPA_STRONG_DEFAULT_KWARGS` |
| `BATCH_DEFAULTS` | `src/treepo/_research/core/batch_transport.py` module-level constants |
| `CONCAT_RATIO`, `DEFAULT_*` | `src/treepo/_research/tasks/manifesto/pipeline_config.py` module-level constants |

---

## Canonical values

### DSPy / LLM (manifesto f,g)

`DSPyFamilyConfig` field defaults are now paper-canonical out of the box;
scripts rarely need to override anything. The manifesto TOML's `[family]`
section contains only `include_identity_targets = true` (required for the
chunked-manifesto artifact; default is `False` because most artifacts have
per-leaf teacher summaries).

| Field | Default | Why this is canonical |
|---|---|---|
| `optimizer` | `"gepa"` | Matches `OptimizationConfig.optimizer_type`. |
| `budget` | `"heavy"` | Matches `OptimizationConfig.gepa_auto`. |
| `lm_context_window_tokens` | `32000` | Matches production Gemma-4-31B-IT-NVFP4 vLLM `--max-model-len 32768`. |
| `max_completion_tokens` | `1024` | Satisfies the two-leaf concat invariant `≥ 2 × leaf_size_tokens=512`. |
| `num_threads` | `128` | Saturates 4-GPU vLLM at typical val-pool sizes. |
| `batch_size` / `batch_max_concurrent` | `64` / `512` | Mirrors `src/treepo/_research/core/batch_transport.py`. |
| `batch_timeout` / `batch_routing_policy` | `0.02` / `"affinity_load_aware"` | Same. |
| `gepa_kwargs` (field default factory) | `dict(GEPA_STRONG_DEFAULT_KWARGS)` | Sourced from `src/treepo/_research/training/gepa_defaults.py::GEPA_STRONG_DEFAULT_KWARGS` — the single lightweight source. |

`DSPyFamily._build_optimizer` reads `self.config.gepa_kwargs` and layers
per-call kwargs (`metric`, `reflection_lm`, `auto`, `num_threads`) on
top. `GEPAOptimizer._build_gepa_kwargs` (in `src/treepo/_research/training/optimization/gepa.py`)
seeds from the same constant. **No monkey-patch, no `apply_X()` setup
call anywhere.**

### FNO family

Loaded directly from upstream `FNOFamilyConfig`. Paper defaults:
`hidden_channels=32`, `n_modes=64`, `n_layers=2`,
`epochs_per_iteration=8`, `batch_size=2`, `learning_rate=1e-3`,
`leaf_size_tokens=512`, `embedding_max_length_tokens=2048`,
`effective_embedding_dim=768`.

Smoke TOML overrides several for fast execution. Note: the **unified-g**
defaults `state_dim=128, hidden_dim=512` live deeper in
`CleanUnifiedNO` and are pinned by
`feedback_head_capacity_was_not_the_bottleneck.md`. Don't widen.

### Markov change-point oracle

DGP loaded directly from upstream `MarkovChangepointConfig`:
`n_regimes=4`, `vocab_size=96`, `min/max_tokens=96`,
`min/max_segments=2/5`, `min/max_seg_len=8/32`, `train_docs=120`,
`test_docs=60`, `sinkhorn_iters=30`, `transition_log_std=1.25`, `seed=0`.
The dispatcher auto-builds eval trees via
`_make_oracle_fixture_markov`; callers just pass `oracle_name` and
optional knob overrides.

### Research-only LDA

LDA leaf-local-mixture and tree-recovery experiments are no longer part of
the public `treepo.methods` examples/configs. Their configs and scripts live
under `configs/research/`, `examples/research/`, `scripts/research/`, and
`src/treepo/_research/`.

### Research-only HLL sketch

The HLL classical-sketch method example is no longer part of
`treepo.methods`. Its config and helper example live under
`configs/research/methods/hll_sketch.toml` and
`examples/research/methods/run_hll_sketch.py`.

### Research-only Markov FNO probe

The standalone `CleanUnifiedNO` Markov probe is no longer part of
`treepo.methods`. Its script and config live under
`scripts/research/probe_clean_unified_no.py` and
`configs/research/methods/markov_probe.toml`, with the helper example at
`examples/research/methods/run_markov_probe.py`.

---

## Findings (what justifies these defaults)

### DSPy / LLM (from `docs/gepa_optimization_handoff_2026-04-21.md`)

1. **Pick the right GEPA scope (40× speedup).** v1 ran GEPA over the full
   pipeline (~12 LM calls per rollout, 21-75h ETA). v2 scoped it to the
   scorer-only on cached summaries (1 LM call/rollout, 30-40 min for the
   light budget).
2. **Rank metric beats MAE.** MAE pushed predictions toward the label
   center, hurting Pearson. Rank metric preserves order.
3. **Baseline guard is mandatory.** If optimized test r < baseline test
   r, keep the baseline. Without it, GEPA can land worse-than-baseline
   programs on validation noise.

### DSPy / LLM (from `docs/manifesto_optimization_writeup.md`)

1. **Leaf-size invariance.** Per-dim external Pearson within 0.014–0.045
   across a 32× leaf-size sweep on 5 of 6 dimensions.
2. **f/g tension on joint metric.** Decentralization oscillates
   0.361 → 0.461 → 0.343 under joint 6-dim training.
3. **Single-dim escape converges and beats Benoit.** Decentralization
   alone reaches Pearson 0.557 (stable). Beats Benoit's proprietary
   18-score ensemble (0.490) at ≈ 1/8 the context budget.
4. **Train pool sizing matters.** 3 train trees → 9 g records → ~4 LLM
   calls/iter; the full multi-leaf split=train pool (14 trees) yields
   ~426 g records. The doc-length filter only matters for the k=0
   raw_concat baseline eval; don't apply it to the train pool.

### Markov + HLL grids (covered by `tests/methods/reproduction/`)

1. **Markov grid: 12/12 cells MAE=0 bit-for-bit** (3 seeds × 2 n_regimes
   × 2 max_tokens).
2. **HLL precision scaling**: MAE decreases monotonically; at p=14,
   MAE = 0.467 (< 5% of mean exact count).
3. **HLL schedule invariance**: `balanced` / `left_to_right` /
   `right_to_left` bit-for-bit identical.

### FNO + research Markov probe (covered by `tests/methods/integration/`)

1. FNO live training step completes on CUDA in 6.6s on RTX PRO 6000
   Blackwell (tiny config). Per-tree predictions are finite floats.
2. Markov FNO probe runs the research script unchanged in 7.8s.
3. **Unified-g head capacity** (`state_dim=128, hidden_dim=512`) is
   conservative-and-correct; widening to 2048/2048/4096 broke
   composition cells. See `feedback_head_capacity_was_not_the_bottleneck.md`.

---

## Single source of truth — no more imperative setup

The cross-family constants in `canonical_defaults.py`
(`BATCH_DEFAULTS`, `CONCAT_RATIO`, `DEFAULT_*`, `GEPA_STRONG_DEFAULTS`)
are direct `import`-redirects of their upstream counterparts, not
hand-written mirrors — drift is structurally impossible. The
`test_cross_family_constants_are_upstream_objects` drift test verifies
this via `is`-identity.

Strong GEPA kwargs live as field defaults on both
`OptimizationConfig.gepa_kwargs` (via `_build_gepa_kwargs` seeding) and
`DSPyFamilyConfig.gepa_kwargs`, both sourced from the same
`GEPA_STRONG_DEFAULT_KWARGS` constant in
`src/treepo/_research/training/gepa_defaults.py`. No monkey-patch, no
`apply_strong_gepa_defaults()` setup call.

The Markov oracle dispatcher auto-builds its corpus via
`_make_oracle_fixture_markov` from oracle config knobs (same shape as
the HLL fixture). No mirror dataclasses are needed.

Every paper-canonical setting is now reachable as either a dataclass
field default or a registered fixture builder. **There is no remaining
imperative setup call in `canonical_defaults.py`.**
