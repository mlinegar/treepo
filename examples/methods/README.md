# `treepo.methods` Example

This package keeps small method examples here:

| Example | What it uses |
|---|---|
| `run_hll_sketch.py` | DataSketches HLL fixture + classical_sketch family through run("fit", ...). |
| `run_fno_markov.py` | Built-in neural_operator family with operator_kind=fno + built-in Markov changepoint fixture. |
| `run_neural_operator_markov_compare.py` | Compare official dense `neuralop` operator kinds on the same Markov fixture. |
| `run_neural_operator_markov_leaf_grid.py` | Leaf-size grid for learned neural operators on the Markov fixture. |
| `run_neural_operator_lda.py` | Overlapping-topic Dirichlet LDA fixture + official sklearn baseline + built-in `neural_operator` family for full topic-proportion vectors. |
| `run_neural_operator_lda_leaf_grid.py` | Leaf-size grid for learned neural operators on the overlapping-topic LDA fixture. |
| `run_manifesto_replications.py` | Central Manifesto/RILE replication shape: root document labels plus qsentence guidance for `g`, runnable with `dspy` or prompted-LLM estimators. |

TRL, diffusion/dgemma, and specialized large-training studies belong with the
packages that own those application layers. DSPy and prompted-LLM examples are
kept provider-neutral here and accept injected programs/callables from downstream
code.

Leaf-grid examples use `treepo.methods.grid` for cell enumeration and JSON/CSV output.

## Estimator Axis

`treepo.methods` separates the estimator used for learned artifacts from the
objective terms used to train or certify them. The default estimator target is
`g`, so `estimator = { name = "fno" }` trains unified `g` through the FNO route.
Built-in estimator values include `neural_operator`, `fno`, `conv1d`, `llm`,
`prompted_llm`, and `dspy`. The LLM/DSPy families are provider-neutral and
accept injected `predict_fn`, `program`, or `dspy_program` hooks. The older
`g_estimator` key is accepted as a compatibility alias. Local-law penalties
remain objective terms, not estimator families.
