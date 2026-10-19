# treepo

**Auditable tree summarization for long documents.** `treepo` is the Python
package behind **C-TreePO** (Compression-Tree Preference Optimization), from
the paper *What Makes a Good Summary? Learning Sufficient Representations at
Scale* (Mitchell Linegar; [paper](https://mlinegar.github.io/research.html)).

## The idea

Some judgments depend on evidence spread across a document too long for one
model call. A party manifesto's left-right position is one example. C-TreePO
splits the document into leaves, summarizes each leaf, merges summaries
pairwise up a binary tree, and judges the root. A summary is *good* when it is
sufficient: the judge gives the same answer from the summary as from the text
it replaces. The paper reduces this to three local laws, each of which can be
checked at a single node:

| Law | Checked at | Holds when | Name in `treepo` |
|---|---|---|---|
| C1 leaf sufficiency | each leaf | judging the leaf's summary gives the same answer as judging the leaf | `leaf_preservation` |
| C2 idempotence | each summary | summarizing a summary again does not change its judgment | `on_range_idempotence` |
| C3 merge consistency | each merge | judging the merged summary gives the same answer as judging its two children | `merge_preservation` |

Because each law is local, you can audit the pipeline without labeling every
node. A trusted judge (a person or a stronger model) labels a random sample of
nodes with known inclusion probabilities. Design-based supervised learning
then combines a cheap proxy check on every node with those sampled labels to
give unbiased estimates of each law's violation rate across the whole tree.
The paper shows how guarantees on the local laws carry up to the root, and
gives design-based confidence envelopes for the realized tree.

`treepo` provides:

- **One learning entry point.** `treepo.fit(...)` trains and evaluates a
  summarizer `g` and a judge `f` across several families: exact oracles,
  classical sketches, neural operators, prompted LLMs, and DSPy programs.
- **Local-law audits.** `LocalLawAuditRow`, node-audit sampling designs, and
  corrected estimates (`treepo.local_law`, `treepo.sampling`), plus per-run
  evidence and error-certificate artifacts.
- **Data contracts.** `TreeRecord`, `TaskState`, and `PreferenceDataset` hold
  trees, node states, and supervision. They export to SFT, DPO, reward-model,
  and GRPO trainer formats.
- **Tree visualization.** `treepo.viz` writes a set of trees to one
  self-contained HTML file that shows sampled nodes and per-node law losses.

## Install

You need Python 3.10 or later. The core package depends only on numpy. The
project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mlinegar/treepo
cd treepo
uv run --no-dev python examples/quickstart.py   # core install into .venv, then run the quickstart
```

`uv sync` installs the full development environment, which includes every
extra and pytest. To depend on `treepo` from another uv project:

```bash
uv add "treepo @ git+https://github.com/mlinegar/treepo"
uv add "treepo[llm] @ git+https://github.com/mlinegar/treepo"   # with an extra
```

| Extra | Adds |
|---|---|
| `torch` | neural-operator families (`neural_operator`, `fno`) |
| `lda` | scikit-learn LDA baseline (`sklearn_lda`) used by the LDA examples |
| `sketches` | DataSketches backends for `classical_sketch` |
| `llm` | HTTP client for OpenAI-compatible servers (`api_base`) |
| `hf` | Hugging Face `datasets` export of preference data |
| `bench` | YAML configs for the `treepo-bench` CLI |
| `all` | everything above |

If an extra is missing, `treepo` raises an error that names it the first time
the extra is needed.

## Quickstart: audit a lossy summarizer

This example runs offline in about a second. It needs no model, API key, or
GPU. Each toy document has eight leaves of eight words. The judge scores net
sentiment, and the summarizer keeps at most four sentiment words. A cheap
proxy check flags every node where the summarizer had to drop words. Trusted
labels go to a random 20% of the node checks.

```python
import random

from treepo.local_law import LocalLawAuditRow, audit_local_laws
from treepo.sampling import apply_node_audit, sample_node_audit
from treepo.schedule import fold_with_trace, merge_children

SENTIMENT = {"good": 1, "great": 1, "win": 1, "bad": -1, "poor": -1, "loss": -1}
BUDGET = 4  # the summarizer keeps at most this many sentiment words


def judge(text):  # f*: the trusted judge (here: net sentiment)
    return sum(SENTIMENT.get(word, 0) for word in text.split())


def summarize(text):  # g: keep sentiment words, up to the budget
    return " ".join([w for w in text.split() if w in SENTIMENT][:BUDGET])


def merge(left, right):  # pairwise merge: re-summarize the two child summaries
    return summarize(left + " " + right)


def proxy(text):  # cheap check: 1.0 if g had to drop anything
    return float(sum(w in SENTIMENT for w in text.split()) > BUDGET)


rng = random.Random(0)
vocab = list(SENTIMENT) + "the a report said vote party plan today".split()
docs = [[" ".join(rng.choices(vocab, k=8)) for _ in range(8)] for _ in range(100)]

rows = []
for d, leaves in enumerate(docs):
    nodes = fold_with_trace([summarize(t) for t in leaves], merge)  # 8 leaves, 7 merges
    children = merge_children(len(leaves))
    for i, summary in enumerate(nodes):
        if i < len(leaves):  # C1: compare with the raw leaf text
            law, source = "c1", leaves[i]
        else:  # C3: compare with the two child summaries
            law, source = "c3", " ".join(nodes[c] for c in children[i])
        rows.append(LocalLawAuditRow(
            row_id=f"{d}:{i}:{law}", law_kind=law, proxy_loss=proxy(source),
            oracle_loss=float(judge(summary) != judge(source)), observed=True))
        rows.append(LocalLawAuditRow(  # C2 on every node
            row_id=f"{d}:{i}:c2", law_kind="c2", proxy_loss=0.0,
            oracle_loss=float(judge(summarize(summary)) != judge(summary)), observed=True))

truth = audit_local_laws(rows)  # toy only: every check has a trusted label
design = sample_node_audit(len(rows), policy="fraction", fraction=0.2, seed=0)
audit = audit_local_laws(apply_node_audit(rows, design))  # trusted labels on 20%

print(f"{'law':22} {'true':>6} {'proxy':>6} {'audit':>6}  labeled")
for law, result in audit["by_law_kind"].items():
    stats = result["local_law_objective"]
    true_rate = truth["by_law_kind"][law]["local_law_objective"]["objective"]
    proxy_rate = sum(r.proxy_loss for r in rows if r.law_kind == law) / stats["row_count"]
    print(f"{law:22} {true_rate:6.3f} {proxy_rate:6.3f} {stats['objective']:6.3f}  "
          f"{stats['observed_count']}/{stats['row_count']}")
```

```text
law                      true  proxy  audit  labeled
leaf_preservation       0.190  0.205  0.193  162/800
merge_preservation      0.671  0.929  0.671  132/700
on_range_idempotence    0.000  0.000  0.000  306/1500
```

The columns are:

- `true`: the violation rate computed from a trusted label on every check.
  Only a toy example can afford this.
- `proxy`: the rate from the cheap check alone.
- `audit`: `treepo`'s design-based estimate, which uses trusted labels on the
  20% sample.

The audit shows where this summarizer breaks. Leaf summaries are usually
sufficient, and summarizing a summary again changes nothing. But two-thirds of
merges change the sentiment score, because the four-word budget binds more
often as summaries move up the tree. The proxy alone overstates merge failures
(0.93), because dropping words does not always change the net score. The
sampled labels correct that bias. Across 500 random 20% draws, the merge
estimate averages 0.672 with a standard deviation of 0.038. Seed 0 happens to
land on the true rate.

The same code is in [`examples/quickstart.py`](examples/quickstart.py). In a
real pipeline, `summarize` and `merge` are LLM calls, `judge` is the scoring
model or a human, and the audit rows have the same shape.

## Fitting `f` and `g` with `treepo.fit`

Every family uses the same call shape. The example below runs with the core
install. It uses the exact oracle on synthetic Markov-changepoint documents as
a reference point:

```python
import treepo
from treepo.methods.fixtures import make_markov_changepoint_trees

trees = make_markov_changepoint_trees(n_trees=8, seed=0)  # synthetic docs with exact labels
result = treepo.fit({
    "family": "oracle",
    "eval_data": trees,
    "backend_config": {"oracle_name": "markov_changepoint_count"},
})
print(result.status, result.metrics["test_internal_f_mae"])  # success 0.0
print(result.summary["statistic_local_law"]["row_count"])     # 88 local-law checks
```

`fit` writes `results.json`, `evidence.json`, and prediction rows under
`outputs/treepo_fit`. Set `backend_config["output_dir"]` to write them
somewhere else. `result.artifacts` lists the files. To learn `g` and `f`
instead, change the family:

| `family` | What it fits | Needs |
|---|---|---|
| `oracle` | exact reference statistics (`markov_changepoint_count`, `hll_exact`) | core |
| `learnable_constant` | one-parameter inverse-probability-weighted baseline | core |
| `classical_sketch` | HyperLogLog and other sketches as composable statistics | `sketches` |
| `neural_operator`, `fno` | learned leaf encoder, merge, and readout (FNO, TFNO, UNO, Conv1D) | `torch` |
| `sklearn_lda` | scikit-learn LDA baseline on topic-proportion trees | `lda` |
| `llm` | a prompted model behind `api_base` or a Python `predict_fn` | core (`llm` for `api_base`) |
| `dspy` | an injected DSPy program or callable | core plus your DSPy program |

For example, this command trains an FNO summarizer and judge on the Markov
fixture, with root and leaf supervision:

```bash
uv run python examples/methods/run_fno_markov.py --output-dir outputs/fno_markov
```

Downstream packages can add families with
`treepo.methods.families.register_family(name, factory)`.

## LLM judges and the Manifesto example

`family="llm"` scores trees with a prompted model. You can point it at any
OpenAI-compatible server (vLLM, SGLang, or a hosted API) with `api_base`, or
pass any Python callable as `predict_fn`. This version runs offline with a
stub in place of the model:

```python
import treepo
from treepo import TreeNode, TreeRecord


def make_doc(doc_id, leaf_texts, label):
    leaves = [TreeNode(node_id=f"leaf{i}", unit_type="leaf", text=t, parent_id="root")
              for i, t in enumerate(leaf_texts)]
    return TreeRecord(tree_id=doc_id, text=" ".join(leaf_texts), root_label=label,
                      nodes=[*leaves, TreeNode(node_id="root", unit_type="root")])


docs = [make_doc("a", ["good plan", "great win"], 3.0),
        make_doc("b", ["bad vote", "poor plan"], -2.0)]


def predict_fn(*, prompt):  # stand-in for a model call; return text containing a number
    words = prompt.split()
    return str(sum(w in {"good", "great", "win"} for w in words)
               - sum(w in {"bad", "poor", "loss"} for w in words))


result = treepo.fit({
    "family": "llm",
    "train_data": docs,
    "eval_data": docs,
    "backend_config": {
        "predict_fn": predict_fn,
        "prompt_template": "Rate the net sentiment of this text.\n\n{text}\n\nScore:",
    },
})
print(result.status, result.metrics["internal_f_mae"])  # success 0.0
```

To use a served model, install the `llm` extra and replace `predict_fn` with
endpoint settings. This configuration needs a running server, for example
`vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000`:

```python
backend_config = {
    "model": "Qwen/Qwen2.5-7B-Instruct",     # must match an id served at /v1/models
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "max_tokens": 256,
    "temperature": 0.0,
    "prompt_template": "Return only one numeric score.\n\n{text}\n\nScore:",
}
```

[`examples/methods/run_llm_backends.py`](examples/methods/run_llm_backends.py)
runs the OpenAI-compatible, Transformers-style, and DSPy-style backends
against fake servers, so you can check the wiring without a GPU.
[`scripts/start_vllm.sh`](scripts/start_vllm.sh) is a launch helper for vLLM.

The Manifesto/RILE walkthrough runs the paper's application path on a
packaged three-document fixture of CMP-coded quasi-sentences:

```bash
uv run python examples/methods/run_manifesto_end_to_end.py --output-dir outputs/manifesto_end_to_end
```

The walkthrough:

1. samples documents and quasi-sentences with logged propensities,
2. passes root and quasi-sentence supervision to `treepo.fit(...)`,
3. writes `evidence.json`,
4. exports DPO, reward-model, and GRPO data for root-only,
   quasi-sentence-only, and combined supervision.

An oracle predictor stands in for the LLM, so the walkthrough runs offline.
Training at full Manifesto Project scale runs in a downstream workspace. That
workspace registers its families with `treepo` and uses these same records.

## More examples

| Command | Shows | Needs |
|---|---|---|
| `uv run python examples/methods/run_local_law_certificate.py --output-dir outputs/cert` | sampled C1/C2/C3 rows, an audit summary, `evidence.json`, and a certificate ledger | core |
| `uv run python examples/methods/run_tree_visualization.py` | five standalone HTML tree views in `outputs/tree_visualization/` | `torch` |
| `uv run python examples/methods/run_finetune_views.py --output-dir outputs/finetune` | one `PreferenceDataset` exported as embedding, SFT, DPO, reward, and GRPO rows | core |
| `uv run treepo-bench run markov --config examples/bench/markov.yaml --json-out outputs/markov.json --csv-out outputs/markov.csv` | the benchmark CLI | `bench` |

[`examples/README.md`](examples/README.md) lists the rest, including
HyperLogLog sketches, LDA, and neural-operator leaf-count grids.

## Package map

| Module | Contents |
|---|---|
| `treepo.fit` | the single learning entry point (`src/treepo/learning.py`, `treepo.methods`) |
| `treepo.local_law` | C1/C2/C3 audit rows, corrected objectives, and `audit_local_laws` |
| `treepo.sampling` | node-audit designs and sampling propensities |
| `treepo.schedule` | the canonical pairwise merge schedule (`fold`, `merge_children`, `merge_depths`) |
| `treepo.tree`, `treepo.state` | `TreeRecord`, `TreeNode`, `TaskState` |
| `treepo.methods.preference`, `treepo.finetune` | `PreferenceDataset` and trainer exports |
| `treepo.evidence`, `treepo.certificate` | run evidence and the error-certificate ledger |
| `treepo.viz` | HTML tree visualization |

For more detail, see [`docs/architecture.md`](docs/architecture.md),
[`docs/tree_and_sampling.md`](docs/tree_and_sampling.md),
[`docs/preference_data.md`](docs/preference_data.md), and
[`docs/evidence.md`](docs/evidence.md).

## Development

```bash
uv sync                                   # dev environment with every extra
uv run pytest -q                          # three tests skip unless local datasets are configured
uv run treepo-bench check release --json  # packaging and docs hygiene gates
```

## Paper and author

- Paper: Mitchell Linegar, *What Makes a Good Summary? Learning Sufficient
  Representations at Scale*. Available at
  [mlinegar.github.io/research.html](https://mlinegar.github.io/research.html).
- Author: Mitchell Linegar, <https://mlinegar.github.io>

License: PolyForm Noncommercial 2.0.0-pre.2. See [`LICENSE`](LICENSE).
