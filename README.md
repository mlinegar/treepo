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

Add optional OpenAI-compatible client helpers or optimizer-backed DSPy with:

```bash
uv add "treepo[llm] @ git+https://github.com/mlinegar/treepo"
uv add "treepo[dspy] @ git+https://github.com/mlinegar/treepo"
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
`treepo[dspy]` for optimizer-backed prompt programs, `treepo[bench]` for
benchmark config IO, and `treepo[all]` for everything.
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
Transformers pipelines or custom local runtimes. This generic ``llm`` family
is an inference/artifact adapter and does not optimize ``g``. DSPy runs use the
same server settings with an optimizer-backed program family that learns the
joint readout ``f`` and, for nonidentity C-Trees, one shared ``g``.

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

For DSPy prompt tuning, provide a non-disabled optimizer plus ``lm_config``.
The family can construct default vector ``f``/``g`` programs from the declared
target schema, or accept explicit ``f_program`` and ``g_program`` adapters.
``dspy_program`` remains a compatibility alias for ``f_program``; it is not a
replacement for the shared ``g_program``.

```python
result = treepo.fit({
    "family": "dspy",
    "schedule": "fg",
    "g_mode": "learned",
    "oracle_targets": oracle_targets,
    "train_data": train_trees,
    "eval_data": eval_trees,
    "preference_data": preferences,
    "backend_config": {
        "optimizer": "bootstrap_random_search",
        "lm_config": lm_config,
        # Optional overrides; otherwise defaults come from oracle_targets.
        # "f_program": f_program,
        # "g_program": g_program,
    },
})
```

The one compiled ``g`` receives both role-tagged leaf and merge examples and is
reused at every call in ``reduce_g``. Generic ``family="llm"`` remains the
inference-only route.

## Examples

For long-document language tasks, start with the LLM families. `family="llm"`
can call an OpenAI-compatible endpoint directly from `api_base`, or accept an
injected `predict_fn`. `family="dspy"` is the optimizer-backed prompt-program
route for one joint vector ``f`` and one shared ``g``. These local examples exercise the
backend adapter shapes and the preference/optimizer views used by those routes:

```bash
uv run python examples/methods/run_llm_backends.py \
  --output-dir outputs/llm_backends_example
```

```bash
uv run python examples/methods/run_preference_optimizer_views.py \
  --output-dir outputs/preference_optimizer_views_example
```

The package's Semantic-Forest example is organized as two isomorphic family
grids. DSPy and FNO each run the same nine cells:
`K={1,3,57} x {full_doc_direct,ctree_base_summary,ctree_recursive}`
(18 cells total). Learned DSPy execution is the default and requires a JSON or
TOML backend config containing ``optimizer`` and ``lm_config`` (or injected
program adapters for programmatic calls):

Learned cells default to `--max-iterations 3`, the alternating `f -> g -> f`
sequence; identity/fixed-`g` cells elide the `g` slot. The learned DSPy grid
also defaults `f_record_source="generated_when_available"`: the first `f`
pass may use reference states because no learned `g` exists yet, while the
final `f` pass consumes states produced by the current shared `g`. Set
`f_record_source="gold_state"` explicitly in the DSPy backend config for the
reference-state ablation.

```bash
uv run python examples/methods/run_manifesto_semantic_forest_grid.py \
  --family both \
  --dspy-execution learned \
  --dspy-config path/to/dspy_backend.toml \
  --output-dir outputs/manifesto_semantic_forest_grid
```

For a fully local interface smoke, request the deterministic fixture explicitly:

```bash
uv run python examples/methods/run_manifesto_semantic_forest_grid.py \
  --family both \
  --dspy-execution offline_fixture \
  --output-dir outputs/manifesto_semantic_forest_grid_offline
```

Only ``offline_fixture`` substitutes the deterministic oracle program and fixed
analytic ``g``; it runs no DSPy optimizer and must never be reported as the
learned DSPy grid. Both modes use synthetic records, so neither command alone
is publication or Polmeth evidence.

A recursive binary C-Tree has `L >= 1` leaves and exactly `M = L - 1` merge
applications. Identity `g` is restricted to the direct no-composition path;
a binary fold requires an explicit fixed or learned state operator. Model
family and target width remain independent axes. A single document-sized leaf
is already a valid C-Tree and has no merges. The package distinguishes three
derived execution paths:

- `full_doc_direct`: `f(X)`, with identity `g` operationally elided;
- `ctree_base_summary`: `f(g(X))`, with one call to a nonidentity `g` and no
  internal call; and
- `ctree_recursive`: `f(reduce_g(T))` on `L >= 2` leaves, where the same `g`
  is called at leaves and internal nodes.

There is only one `g` artifact and one set of learned parameters. Formally,
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`.
`reduce_g` is this induced fold, not another operator, learner, artifact, or
optimization stage.

Thus `full_doc` names full-span singleton geometry, not `g_mode`, and `ctree`
is the umbrella grammar rather than shorthand for `L >= 2`. A comparison grid
may use `ctree` as a short label for its deliberately multi-leaf arm, but that
is a cell definition rather than the definition of a C-Tree. In the packaged
Semantic-Forest grid, `ctree_recursive` fits one `(f, g)` pair for each
family/target width and `ctree_base_summary` evaluates those exact artifacts
at zero iterations. `full_doc_direct` is a separate direct-readout control,
not a third independently fitted leaf-count view of that pair.

A fixed nonidentity operator must be declared `g_mode="fixed"`; a trainable
operator is called learned only when `g_contract.learned_this_run` is true, or
learned-reused when a verified initial artifact is evaluated without an update.
On a summarized singleton, an update changes the same shared `g` using
leaf-call support only; it supplies no internal-call or C3 evidence for
composition.
`train_g_call_count` records attempted calls separately from `g_update_count`;
downstream families can return `treepo.methods.GTrainOutcome` when artifact
equality cannot disclose whether an update actually occurred.

The provider-neutral families have distinct semantics. ``family="llm"``
supports inference and artifact plumbing but does not optimize ``g``.
``family="dspy"`` is optimizer-backed: it learns vector ``f`` and, when
``g_mode="learned"`` reaches a g-step, compiles one shared ``g`` over the
available leaf/merge call domains. The result calls it learned only when the
realized contract reports an update. ``offline_fixture`` is a fixed analytic
control and is never learned-g evidence.

`K={1,3,57}` all use the same `treepo.fit(...)` API, exact named target-vector
contract, and mean document-level sum-L1 metric; `K=1` is not a scalar-only
execution branch.
The FNO/neural-operator families train root, node-readout, and vector-state
rows with that same unnormalized `sum_l1` reduction by default. Set
`backend_config["training_loss"]="coordinate_mean_mse"` only to reproduce the
legacy coordinate-mean MSE surrogate.

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
[`docs/preference_data.md`](docs/preference_data.md),
[`docs/semantic_forests.md`](docs/semantic_forests.md),
[`docs/full_document_llm_contract.md`](docs/full_document_llm_contract.md), and
[`examples/`](examples/).

License: see [`LICENSE`](LICENSE).
