# Boundary

`treepo` is the v0.1.0 package for the C-TreePO code.

## Included

- Base install: `numpy`, `torch`, and `neuraloperator` for the built-in neural-operator family.
- Public modules: `treepo`, `treepo.methods`, `treepo.local_law`,
  `treepo.bench`, `treepo.llm`, `treepo.training`, and small task fixtures.
- Source examples: `examples/bench` and `examples/methods`.

## Extras

- `bench`: YAML config IO.
- `sketches`: datasketches-backed sketch adapters.
- `torch`: explicit selector for torch-backed helpers.
- `neural`: explicit selector for the built-in neural-operator dependency stack.
- `llm`: OpenAI-compatible and native-Transformers helpers plus
  `langextract`, `tiktoken`, and `requests`.
- `train`: torch local-law training helpers.
- `all`: all optional package stacks.

`uv sync --no-dev` installs the base package. `uv sync` installs contributor
dependencies too.

## Extensions

TRL, diffusion/dgemma, large manifesto-training campaigns, deployment, and
large simulation code belongs in packages or workspaces that register with the
public `treepo` contracts. DSPy and prompted-LLM estimator routes are included
as provider-neutral wrappers; downstream code supplies the actual program or
callable.

Extension stubs preserve public names and provide clear registration errors.
They must not import heavy dependencies. Markov has a built-in changepoint
fixture/oracle task benchmark. LDA and Manifesto/RILE have small native fixtures
for method examples; full-scale workflows can still live downstream.

## Checks

Run:

```bash
uv lock --check
uv run treepo-bench check release --json
uv run pytest -q
uv run python -m treepo.release
uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts
```
