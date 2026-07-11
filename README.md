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

The core install is slim (numpy only). Heavy stacks are extras: `treepo[torch]`
for the neural-operator families (`fno`, `neural_operator`), `treepo[lda]` for
the sklearn LDA family, `treepo[hf]` for Hugging Face dataset export,
`treepo[sketches]` for sketch backends, `treepo[llm]` for LLM clients,
`treepo[bench]` for benchmark config IO, and `treepo[all]` for everything.
Missing extras fail lazily at first use with the extra named in the error.

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

For prompted-LLM runs, install `treepo[llm]`. `family="llm"` can call
OpenAI-compatible `/v1` servers directly, including vLLM, SGLang, hosted
OpenAI-compatible APIs, and other compatible servers. It can also use any
direct Python callable through `predict_fn`, including Hugging Face
Transformers pipelines or custom local runtimes. DSPy runs use the same server
settings plus an injected DSPy program.

One common local setup is:

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct  # replace with your HF model id or local path
SERVED_MODEL_NAME="$MODEL"      # or a short alias; must match lm_config["model"]
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
requires an OpenAI-compatible `/v1` endpoint and a model name that appears in
`/v1/models`. If your vLLM environment ships CUDA libraries outside the system
path, activate that environment before launch; the source checkout also includes
`scripts/start_vllm.sh`, which sets up common bundled CUDA runtimes.

```python
lm_config = {
    # Must match the model id returned by /v1/models, or the alias passed to
    # vLLM with --served-model-name.
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "max_tokens": 256,
    "temperature": 0.0,
}

result = treepo.fit({
    "family": "llm",
    "train_data": train_trees,
    "eval_data": eval_trees,
    "preference_data": preferences,
    "backend_config": {
        **lm_config,
        "prompt_template": "Return only one numeric score.\n\n{text}\n\nScore:",
    },
})
```

For direct local inference with Transformers or another runtime, pass a
callable instead of `api_base`:

```python
def predict_fn(*, prompt, **kwargs):
    output = pipeline(prompt, max_new_tokens=16)
    return output[0]["generated_text"]

result = treepo.fit({
    "family": "llm",
    "train_data": train_trees,
    "eval_data": eval_trees,
    "backend_config": {"predict_fn": predict_fn},
})
```

For DSPy prompt tuning, configure `program` against the same `lm_config` and
pass it through `backend_config`:

```python
result = treepo.fit({
    "family": "dspy",
    "train_data": train_trees,
    "eval_data": eval_trees,
    "preference_data": preferences,
    "backend_config": {"lm_config": lm_config, "dspy_program": program},
})
```

## Examples

For long-document language tasks, start with the LLM families. `family="llm"`
can call an OpenAI-compatible endpoint directly from `api_base`, or accept an
injected `predict_fn`. `family="dspy"` wraps an injected DSPy program or
prediction callable for prompt tuning. These local examples exercise the
backend adapter shapes and the preference/optimizer views used by those routes:

```bash
uv run python examples/methods/run_llm_backends.py \
  --output-dir outputs/llm_backends_example
```

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
