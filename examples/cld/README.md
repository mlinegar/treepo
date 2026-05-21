# `treepo.cld` examples — one per family

Every example follows the same three-step pattern:

1. `load_dataclass(path, cls, section=..., overrides=...)` — TOML → typed config.
   (For Markov probe, just `tomllib.loads(...)` → flat dict.)
2. (Optional) build any scenario-specific things (eval data, lm_config dict).
3. `treepo.cld.run("<method>", {...})` — dispatch.

Where possible, the dataclass IS the upstream truth (`FNOFamilyConfig`,
`MarkovChangepointConfig`, `LDATreeRecoveryConfig`, `DSPyFamilyConfig`).
Where there's no upstream class, `treepo.cld.canonical_defaults` provides
a small one (`HllSketchConfig`, `LdaOracleConfig`). For the Markov probe
(subprocess dispatch with no Python dataclass), the TOML is a flat dict
validated against `allowed_config_keys("probe")`.

| Example | Dispatcher call | TOML | Loaded type(s) |
|---|---|---|---|
| `run_manifesto_fg_compile.py` | `run("fit", {family="dspy", ...})` | `configs/manifesto_fg_compile.toml` | `DSPyFamilyConfig` (upstream) + `LmSection` |
| `run_fno_family.py` | `run("fit", {family="fno", ...})` | `configs/fno_smoke.toml` | `FNOFamilyConfig` (upstream) |
| `run_markov_probe.py` | `run("probe", {...})` | `configs/markov_probe.toml` | flat dict (validated via `allowed_config_keys`) |
| `run_markov_oracle.py` | `run("oracle", {oracle_name="markov_changepoint_count", ...})` | `configs/markov_oracle.toml` | `MarkovChangepointConfig` (upstream) |
| `run_hll_sketch.py` | `run("sketch", {sketch_kind="hll", ...})` | `configs/hll_sketch.toml` | `HllSketchConfig` |
| `run_lda_oracle.py` | `run("oracle", {oracle_name="leaf_local_mixture_target", ...})` | `configs/lda_oracle.toml` | `LdaOracleConfig` |
| `run_lda_recovery.py` | direct `run_lda_tree_recovery_experiment(cfg)` | `configs/lda_recovery_smoke.toml` | `LDATreeRecoveryConfig` (upstream) |

## Adding a new family

1. If the family has an upstream dataclass, point at it directly. If
   not, add a small one to `canonical_defaults.py`.
2. Add a TOML in `configs/`.
3. Copy whichever existing example is closest in shape (~30–60 lines).

That's it. No mirror dataclasses, no `build_*_config_dict` translators,
no per-family drift tests — the parametrized drift fixture grows by one
line.

For canonical values and the empirical history, see
[`../docs/training_defaults.md`](../docs/training_defaults.md).
