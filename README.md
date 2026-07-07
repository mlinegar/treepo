# treepo

`treepo` is a Python package for learning and auditing composable tree
operators for C-TreePO.

It is for tasks where a judgment depends on information spread across a long
document: split the document, compose local summaries, and check whether the
root preserves the task-relevant value. The C-TreePO paper gives local-law
propagation guarantees and sampled-node audits with design-based confidence
envelopes for the realized tree.

It provides a small public API for fitting tree-operator families, recording
tree and preference artifacts, and checking local laws over composable states.

## Setup

Add `treepo` to a `uv` project:

```bash
uv add "treepo @ git+https://github.com/mlinegar/treepo"
```

Add the optional OpenAI-compatible client helpers with:

```bash
uv add "treepo[llm] @ git+https://github.com/mlinegar/treepo"
```

From a source checkout:

```bash
uv sync
uv run pytest -q
uv run treepo-bench run markov \
  --config examples/bench/markov.yaml \
  --json-out outputs/markov.json \
  --csv-out outputs/markov.csv
```

Optional extras are defined in `pyproject.toml` for benchmark config IO, sketch
backends, and LLM clients.

## Usage

Use the Python API with any registered family:

```python
import treepo

result = treepo.fit({
    "family": "oracle",
    "train_data": train_trees,
    "eval_data": eval_trees,
})
```

For DSPy or prompted-LLM runs, start an OpenAI-compatible server for your model
and configure DSPy or your `predict_fn` against its `/v1` endpoint, for example
`http://localhost:8000/v1`. Then pass the configured callable or DSPy program
through `backend_config`. `treepo` supplies prompt, fit, and artifact helpers;
server startup and credentials stay in your application.

One common local setup is:

```bash
MODEL=/path/or/hf-model
SERVED_MODEL_NAME=treepo-local
HOST=0.0.0.0
PORT=8000
TENSOR_PARALLEL=1
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.85
API_KEY=EMPTY

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --api-key "$API_KEY"

curl -H "Authorization: Bearer $API_KEY" \
  "http://localhost:${PORT}/v1/models"
```

Add any model-specific vLLM flags to the `vllm serve` command; `treepo` only
requires an OpenAI-compatible `/v1` endpoint and a matching model name.

```python
lm_config = {
    "model": "treepo-local",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "max_tokens": 256,
    "temperature": 0.0,
}

# Configure `program` with DSPy against `lm_config` before passing it here.
result = treepo.fit({
    "family": "dspy",
    "train_data": train_trees,
    "eval_data": eval_trees,
    "preference_data": preferences,
    "backend_config": {
        "lm_config": lm_config,
        "dspy_program": program,
    },
})
```

## Examples

For long-document language tasks, start with the LLM families. `family="llm"`
renders prompts and accepts a `predict_fn`; `treepo.llm` provides
OpenAI-compatible client helpers. `family="dspy"` wraps an injected DSPy
program or prediction callable for prompt tuning. This local example shows the
preference and optimizer views used by those routes:

```bash
uv run python examples/methods/run_preference_optimizer_views.py \
  --output-dir outputs/preference_optimizer_views_example
```

For numeric fixtures, start with the Markov benchmark:

```bash
uv run treepo-bench run markov \
  --config examples/bench/markov.yaml \
  --json-out outputs/markov.json \
  --csv-out outputs/markov.csv
```

More source-tree examples cover Manifesto/RILE, HyperLogLog sketches,
local-law certificates, visualization, and neural operators (`fno`, `tfno`,
`uno`, `conv1d`). See [`examples/`](examples/).

## Package Map

- Methods and objectives: `treepo.fit(...)`, `treepo.methods`, and
  `treepo.objective.ObjectiveSpec`.
- Gold data and imported datasets: use `TreeRecord`, `TreeNode`, `TaskState`,
  `treepo.tree.load_tree_records`, and `treepo.tree.write_tree_records_jsonl`.
- Labels, preferences, and online annotation outputs: store them as
  `PreferenceDataset` records with candidate scores, ranks, propensities, and
  metadata.
- Audits and guarantees: use `treepo.local_law.LocalLawAuditRow`,
  `treepo.local_law`, `treepo.evidence`, and `treepo.certificate`.
- Trainer exports: use `treepo.finetune` and `treepo.methods.preference` for
  supervised, DPO, reward-model, and GRPO views.
- New implementations: register a family with
  `treepo.methods.families.register_family(...)` or provide an LLM/DSPy
  callable through `backend_config`.

For details, see [`docs/architecture.md`](docs/architecture.md),
[`docs/tree_and_sampling.md`](docs/tree_and_sampling.md),
[`docs/preference_data.md`](docs/preference_data.md), and
[`examples/`](examples/).

License: see [`LICENSE`](LICENSE).
