# `treepo.methods` examples — one per family

Every example follows the same three-step pattern:

1. `load_dataclass(path, cls, section=..., overrides=...)` — TOML → typed config.
2. (Optional) build any scenario-specific things (eval data, lm_config dict).
3. `treepo.methods.run("<method>", {...})` — dispatch.

Where possible, the dataclass IS the upstream truth (`FNOFamilyConfig`,
`MarkovChangepointConfig`, `DSPyFamilyConfig`).
Research-only benchmark configs live under `examples/research/` and
`configs/research/` rather than adding public dispatcher methods.

| Example | Dispatcher call | TOML | Loaded type(s) |
|---|---|---|---|
| `run_manifesto_fg_compile.py` | `run("fit", {family="dspy", ...})` | `configs/research/methods/manifesto_fg_compile.toml` | `DSPyFamilyConfig` (upstream) + `LmSection` |
| `run_fno_family.py` | `run("fit", {family="fno", ...})` | `configs/research/methods/fno_smoke.toml` | `FNOFamilyConfig` (upstream) |
| `run_markov_oracle.py` | `run("oracle", {oracle_name="markov_changepoint_count", ...})` | `configs/research/methods/markov_oracle.toml` | `MarkovChangepointConfig` (upstream) |
LDA method demos, the HLL sketch helper, and the standalone Markov FNO
probe are research-only and live under `examples/research/methods/`.

## Adding a new family

1. If the family has an upstream dataclass, point at it directly. If
   not, add a small one to `canonical_defaults.py`.
2. Add a TOML in `configs/research/methods/`.
3. Copy whichever existing example is closest in shape (~30–60 lines).

That's it. No mirror dataclasses, no `build_*_config_dict` translators,
no per-family drift tests — the parametrized drift fixture grows by one
line.

For canonical values and the empirical history, see
[`../../../docs/training_defaults.md`](../../../docs/training_defaults.md).
