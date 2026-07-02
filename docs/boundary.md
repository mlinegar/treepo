# Boundary

`treepo` is the Python package for C-TreePO.

## Included

- Base install: `numpy`, `scipy`, `scikit-learn`, Hugging Face `datasets`,
  `torch`, and `neuraloperator` for the built-in neural-operator family.
- Public modules: `treepo`, `treepo.methods`, `treepo.local_law`,
  `treepo.state`, `treepo.tree`, `treepo.statistic`, `treepo.artifacts`,
  `treepo.bench`, `treepo.llm`, `treepo.training`, and small task fixtures.
- Generic f/g supervision, JSONable task states, minimal labeled tree records,
  tree validation/summaries, unit-level preference datasets, local-law row
  artifacts, and projection exports for downstream DPO, reward-model, or GRPO
  trainers.
- Source examples: `examples/bench` and `examples/methods`.

## Extras

- `bench`: YAML config IO.
- `sketches`: datasketches-backed sketch adapters.
- `llm`: the requests-based OpenAI-compatible client layer.
- `all`: all optional package stacks.

`uv sync --no-dev` installs the base package. `uv sync` installs contributor
dependencies too.

## Extensions

Trainer applications, manifesto-training campaigns, deployment, and large
simulation code register with the public `treepo` contracts from their own
packages. DSPy and prompted-LLM families are included as provider-neutral
wrappers; downstream code supplies the actual program or callable. Preference
data export is included; policy optimization engines consume those records
downstream.

Additional family runtimes register with
`treepo.methods.families.register_family(...)`. Markov has a built-in
changepoint fixture/oracle task benchmark. LDA and Manifesto/RILE have small
native fixtures for method examples; full-scale workflows register from
downstream packages.

## Checks

Run:

```bash
uv lock --check
uv run treepo-bench check release --json
uv run pytest -q
uv run python -m treepo.release
uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts
```
