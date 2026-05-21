# `treepo.cld` paper-canonical defaults

One pattern across every family. No family has special logic the others don't.

## The pattern (three lines, every family)

```python
from treepo.cld.canonical_defaults import load_dataclass
from <upstream_or_canonical_module> import <ConfigDataclass>

cfg = load_dataclass("treepo/configs/cld/<family>.toml", <ConfigDataclass>)
treepo.cld.run("<method>", {"backend_config": {..., "<thing>_config": cfg}, ...})
```

If a family has an **upstream dataclass** (FNO, DSPy, Markov DGP, LDA
recovery), the TOML loads directly into it — no mirror, no translator.
If a family doesn't (HLL, LDA oracle, Markov probe), the canonical
dataclass lives in `treepo.cld.canonical_defaults` (small, ~5-20 fields).

## Families covered

| Family | Method | Loaded type | TOML | Example |
|---|---|---|---|---|
| **DSPy / LLM** | `run("fit", {family="dspy", ...})` | `DSPyFamilyConfig` *(upstream)* + `LmSection` | [`configs/manifesto_fg_compile.toml`](../configs/manifesto_fg_compile.toml) | [`examples/run_manifesto_fg_compile.py`](../examples/run_manifesto_fg_compile.py) |
| **FNO family** | `run("fit", {family="fno", ...})` | `FNOFamilyConfig` *(upstream)* | [`configs/fno_smoke.toml`](../configs/fno_smoke.toml) | [`examples/run_fno_family.py`](../examples/run_fno_family.py) |
| **Markov FNO probe** | `run("probe", {...})` | flat dict-of-knobs (validated by `allowed_config_keys("probe")`) | [`configs/markov_probe.toml`](../configs/markov_probe.toml) | [`examples/run_markov_probe.py`](../examples/run_markov_probe.py) |
| **Markov change-point oracle** | `run("oracle", {oracle_name="markov_changepoint_count", ...})` | `MarkovChangepointConfig` *(upstream)* | [`configs/markov_oracle.toml`](../configs/markov_oracle.toml) | [`examples/run_markov_oracle.py`](../examples/run_markov_oracle.py) |
| **HLL classical sketch** | `run("sketch", {sketch_kind="hll", ...})` | `HllSketchConfig` | [`configs/hll_sketch.toml`](../configs/hll_sketch.toml) | [`examples/run_hll_sketch.py`](../examples/run_hll_sketch.py) |
| **LDA leaf-local-mixture oracle** | `run("oracle", {oracle_name="leaf_local_mixture_target", ...})` | `LdaOracleConfig` | [`configs/lda_oracle.toml`](../configs/lda_oracle.toml) | [`examples/run_lda_oracle.py`](../examples/run_lda_oracle.py) |
| **LDA tree-recovery** | direct `run_lda_tree_recovery_experiment(cfg)` | `LDATreeRecoveryConfig` *(upstream)* | [`configs/lda_recovery_smoke.toml`](../configs/lda_recovery_smoke.toml) | [`examples/run_lda_recovery.py`](../examples/run_lda_recovery.py) |

*Upstream* = the dataclass already exists in the main repo (`src/...`).
The TOML loads into it directly. No mirror code in `treepo.cld`.
For Markov probe specifically, the upstream "truth" is the probe script's
argparse defaults (`scripts/probe_clean_unified_no.py`); the TOML is a
flat dict forwarded as `--flag` arguments, with keys validated against
`allowed_config_keys("probe")` in `methods.py`.

## Adding a new family

Cost: one TOML + a ~30-line example script. If the family has no upstream
dataclass, also add a small one to `canonical_defaults.py` (~10 lines).

That's the entire surface. No `RunConfig` wrappers, no
`build_*_config_dict` translators, no per-family drift tests. The
parametrized drift test grows by one line per family.

---

## Sources of truth

| Family | Upstream truth (loaded directly) |
|---|---|
| DSPy / LLM family | `src/ctreepo/dspy_family.py::DSPyFamilyConfig` |
| FNO family | `src/ctreepo/fno_family.py::FNOFamilyConfig` |
| Markov change-point DGP | `src/tree/markov_changepoint_honesty_simulation.py::MarkovChangepointConfig` |
| LDA tree-recovery | `src/ctreepo/sim/core/lda_tree_recovery.py::LDATreeRecoveryConfig` |

Cross-family constants (mirrored in `canonical_defaults.py` and pinned
by drift tests):

| Constant | Upstream |
|---|---|
| `GEPA_STRONG_DEFAULTS` | `src/training/config.py::OptimizationConfig` field defaults |
| `BATCH_DEFAULTS` | `src/core/batch_transport.py` module-level constants |
| `CONCAT_RATIO`, `DEFAULT_*` | `src/tasks/manifesto/pipeline_config.py` module-level constants |

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
| `batch_size` / `batch_max_concurrent` | `64` / `512` | Mirrors `src/core/batch_transport.py`. |
| `batch_timeout` / `batch_routing_policy` | `0.02` / `"affinity_load_aware"` | Same. |
| `gepa_kwargs` (field default factory) | `dict(GEPA_STRONG_DEFAULT_KWARGS)` | Sourced from `src/training/optimization/gepa.py::GEPA_STRONG_DEFAULT_KWARGS` — the single upstream source. |

`DSPyFamily._build_optimizer` reads `self.config.gepa_kwargs` and layers
per-call kwargs (`metric`, `reflection_lm`, `auto`, `num_threads`) on
top. `GEPAOptimizer._build_gepa_kwargs` (in `src/training/optimization/gepa.py`)
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

### Markov FNO probe

TOML is a flat dict-of-knobs (no dataclass mirror); the dispatcher
validates keys against `allowed_config_keys("probe")` in
`treepo/src/treepo/cld/methods.py` and forwards each as a `--flag`
to `scripts/probe_clean_unified_no.py`. Anything omitted falls back to
the probe script's own argparse default. Common knobs:
`benchmark="recoverable_v5_t2048"`, `leaf_tokens=2048`,
`train_docs=1024`, `epochs=30`, `batch_size=16`, `channels=64`,
`g_n_modes=32`, `g_n_layers=2`, `scorer_n_modes=16`,
`scorer_n_layers=2`, `lr=1e-4`, `optimizer="adamw"`,
`lr_schedule="cosine"`, `grad_clip=1.0`, `leaf_pool="sum"`,
`training_objective="root"`. The drift test
`test_probe_allowed_keys_cover_probe_argparse` ensures
`allowed_config_keys` stays a superset of the probe's actual argparse
surface (fails if the probe gains a new `--flag` we haven't listed).

### Markov change-point oracle

DGP loaded directly from upstream `MarkovChangepointConfig`:
`n_regimes=4`, `vocab_size=96`, `min/max_tokens=96`,
`min/max_segments=2/5`, `min/max_seg_len=8/32`, `train_docs=120`,
`test_docs=60`, `sinkhorn_iters=30`, `transition_log_std=1.25`, `seed=0`.
The dispatcher auto-builds eval trees via
`_make_oracle_fixture_markov`; callers just pass `oracle_name` and
optional knob overrides.

### HLL classical sketch

Loaded into `HllSketchConfig`: `backend="native"`, `precision=14` (min-MAE
point in the paper grid), `hash_bits=64`, `schedule="balanced"` (HLL is
schedule-invariant), fixture knobs `n_trees=6`, `leaves_per_tree=4`,
`leaf_token_count=24`, `vocabulary_size=200`, `seed=0`.

### LDA leaf-local-mixture oracle

Loaded into `LdaOracleConfig`: `oracle_name="leaf_local_mixture_target"`,
`n_trees=8`, `seed=0`, `split="test"`. The LDA oracle has no v1
auto-fixture; the example builds eval trees via
`LDATreeRecoveryConfig`.

### LDA tree-recovery

Loaded directly from upstream `LDATreeRecoveryConfig`: `n_topics=8`,
`vocab_size=512`, `min/max_tokens=384`, `leaf_tokens=16`, `train_docs=0`,
`test_docs=1024`, `seed=0`. Smoke TOML overrides for fast tests.

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

### Markov + HLL grids (from `docs/treepo_cld_reproduction.md`)

1. **Markov grid: 12/12 cells MAE=0 bit-for-bit** (3 seeds × 2 n_regimes
   × 2 max_tokens).
2. **HLL precision scaling**: MAE decreases monotonically; at p=14,
   MAE = 0.467 (< 5% of mean exact count).
3. **HLL schedule invariance**: `balanced` / `left_to_right` /
   `right_to_left` bit-for-bit identical.

### FNO + Markov probe (from `docs/treepo_cld_reproduction.md`)

1. FNO live training step completes on CUDA in 6.6s on RTX PRO 6000
   Blackwell (tiny config). Per-tree predictions are finite floats.
2. Markov FNO probe runs the paper script unchanged in 7.8s.
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
`src/training/optimization/gepa.py`. No monkey-patch, no
`apply_strong_gepa_defaults()` setup call.

The Markov oracle dispatcher auto-builds its corpus via
`_make_oracle_fixture_markov` from oracle config knobs (same shape as
HLL and LDA fixtures). The probe accepts a flat dict of knobs validated
against `allowed_config_keys("probe")`. No mirror dataclasses for either.

Every paper-canonical setting is now reachable as either a dataclass
field default or a registered fixture builder. **There is no remaining
imperative setup call in `canonical_defaults.py`.**
