# `treepo.methods` Examples

This package keeps small method examples here:

Each runnable `run_*.py` file is written as a vignette: load a paired config or
small fixture, call the public package functions, and write a compact result.
One-off fixture construction, toy records, baseline calculations, and backend
setup live in `example_setup/` so the runnable files stay readable.

| Example | What it uses |
|---|---|
| `run_hll_sketch.py` | DataSketches HLL fixture + classical_sketch family through `treepo.fit(...)`. |
| `run_fno_markov.py` | Built-in neural_operator family with operator_kind=fno and root/leaf supervision through `PreferenceDataset`. |
| `run_neural_operator_markov_compare.py` | Compare official dense `neuralop` operator kinds on the same Markov fixture. |
| `run_neural_operator_markov_leaf_grid.py` | Leaf-grouping grid for learned neural operators on the Markov fixture. |
| `run_neural_operator_lda.py` | Overlapping-topic Dirichlet LDA fixture + official sklearn baseline + built-in `neural_operator` family for full topic-proportion vectors. |
| `run_neural_operator_lda_leaf_grid.py` | Leaf-grouping grid for learned neural operators on the overlapping-topic LDA fixture. |
| `run_manifesto_end_to_end.py` | Full Manifesto/RILE package walkthrough: sampled docs/qsentences, `treepo.fit(...)`, evidence JSON, and root/qsentence/both reward exports. |
| `run_manifesto_replications.py` | Central Manifesto/RILE replication shape: root document labels, sampled document-unit labels, and optional preference exports for DPO/reward/GRPO. |
| `run_manifesto_reward_mechanisms.py` | Trainer-neutral Manifesto preference exports for root-only, qsentence-only, and combined DPO/reward/GRPO views. |
| `run_finetune_views.py` | Task-neutral fine-tuning export skeleton: one `PreferenceDataset` feeds embedding pairs/triplets/ranked rows plus SFT/DPO/reward/GRPO rows. |
| `run_manifesto_finetune_views.py` | Manifesto/RILE fine-tuning exports: root `f` rows, qsentence `g` rows, and qsentence pairwise/ranked candidate views. |
| `run_preference_optimizer_views.py` | Task-neutral optimizer-view skeleton: one `PreferenceDataset` feeds supervised DSPy prompts plus DPO/reward/GRPO projections. |
| `run_local_law_certificate.py` | Minimal sampled C1/C2/C3 audit rows, preference exports, evidence JSON, and component-radius certificate ledger. |
| `run_tree_visualization.py` | Standalone expandable-tree HTML views: Manifesto sampling + gold labels + policy summaries; Markov audited local-law losses, node readouts, and the AIPW audit panel on the synthesized merge tree; LDA readouts vs exact topic proportions; HLL exact distinct counts; and hand-built generic records. See [`docs/visualization.md`](../../docs/visualization.md). |

Preference records can be passed to `treepo.fit({"preference_data": ...})` and
are exported through supervised, DPO, reward-model, and GRPO views.
`PreferenceDataset` also writes a Hugging Face `DatasetDict` with `units` and
`candidates` tables for downstream trainers. Specialized training studies
belong with the packages that own those application layers. DSPy and
prompted-LLM examples are kept provider-neutral here and accept injected
programs/callables from downstream code.

`run_manifesto_replications.py` accepts `preference_mode = "none" | "scores" |
"pairwise" | "ranked"` and `preference_scope = "both" | "roots" |
"qsentences"` in its TOML config.

For the complete packaged path, use `run_manifesto_end_to_end.py`:

```bash
uv run python examples/methods/run_manifesto_end_to_end.py \
  --output-dir outputs/manifesto_end_to_end_example
```

The default run uses every packaged training document and qsentence, passes
combined root/qsentence score supervision to `treepo.fit(...)`, writes
`evidence.json`, writes a canonical artifact bundle with Manifesto `TreeRecord`s,
and exports root-only, qsentence-only, and combined DPO, reward-model, and
GRPO views. The result JSON also includes compact optimizer-view previews for
the fit preferences and each reward cell.

## Manifesto Supervision Grid

A document has atomic units named by `doc_unit_kind`; a leaf groups `leaf_unit_count` of those units. For Manifesto the document unit is `qsentence`, while Markov/LDA use `token` and HLL uses `item`.

The manifesto example supports the two intended supervision lanes.

Root-only cells use document-level RILE labels to train/evaluate `f` and leave
`preference_data` unset:

```toml
preference_mode = "none"
doc_unit_kind = "qsentence"
leaf_unit_counts = [1, 2]
supervision_grid = ["none"]
```

Unit-supervised cells add sampled gold labels as structured
`TaskState(kind="manifesto_policy")` targets for `g`. In this fixture the unit
kind is `qsentence`; downstream tasks can use the same shape for paragraphs,
sections, or extractor spans:

```toml
doc_unit_kind = "qsentence"
leaf_unit_counts = [1]
supervision_grid = ["scores"] # or "pairwise" / "ranked"
doc_sample_size = 2            # optional document sampling
qsentence_sample_size = 1      # optional qsentence sampling per sampled doc
```

The Manifesto example logs known design propensities in the DSL style. Each
cell writes `sampling/document_sampling_rows.jsonl` for the document population
and, when document-unit supervision is enabled,
`sampling/qsentence_sampling_rows.jsonl` for observed and unobserved
qsentences within sampled documents. Exported preference units use the joint propensity for
`PreferenceRecord.propensity`; the component values
`document_propensity`, `unit_propensity`, and `label_propensity` are preserved in
unit metadata.

The ordinary tests cover the full packaged fixture. A skipped-by-default
integration test checks the real local Manifesto Project CSVs when a local
Manifesto Project CSV checkout is present:

```bash
TREEPO_RUN_MANIFESTO_PROJECT_FULL=1 \
  uv run pytest tests/methods/test_manifesto_project_full_integration.py -q
```

For task-neutral fine-tuning exports, use `run_finetune_views.py`:

```bash
uv run python examples/methods/run_finetune_views.py \
  --output-dir outputs/finetune_views_example
```

The example writes embedding pairs, embedding triplets, embedding ranked groups,
SFT rows, and pass-through DPO/reward/GRPO rows from one `PreferenceDataset`.
It is export-only; downstream packages can run sentence-transformers or TRL
against the generated files.

For Manifesto/RILE fine-tuning exports, use `run_manifesto_finetune_views.py`:

```bash
uv run python examples/methods/run_manifesto_finetune_views.py \
  --output-dir outputs/manifesto_finetune_views_example
```

The Manifesto example writes root/document labels as `f` SFT rows, qsentence
policy states as `g` SFT rows, and qsentence pairwise/ranked candidate rows for
embedding and TRL-compatible workflows.

For a task-neutral optimizer/DSPy skeleton, use `run_preference_optimizer_views.py`:

```bash
uv run python examples/methods/run_preference_optimizer_views.py \
  --output-dir outputs/preference_optimizer_views_example
```

The example uses one `PreferenceDataset` to write supervised, DPO, reward-model,
and GRPO rows, then passes that same dataset to `treepo.fit(...)` with
`family = "dspy"` and an injected predictor. It runs fully locally; downstream
trainers consume the exported rows. The Manifesto end-to-end example uses the
same pattern on real package fixtures.

For reward-model data, `run_manifesto_reward_mechanisms.py` exports the same
preference records through trainer-specific projections:

```bash
uv run python examples/methods/run_manifesto_reward_mechanisms.py \
  --output-dir outputs/manifesto_reward_mechanisms_example
```

The default grid writes root-only, qsentence-only, and combined scopes for both
pairwise and ranked preferences. Each cell includes `preference_dpo.jsonl`,
`preference_reward.jsonl`, `preference_grpo.json`, the general preference
dataset, and a Hugging Face `DatasetDict`.

A combined grid is also valid. It runs root-only cells at the requested
leaf-grouping sizes and one unit-supervised cell with one document unit per
leaf:

```toml
doc_unit_kind = "qsentence"
leaf_unit_counts = [1, 2]
supervision_grid = ["none", "scores"]
```

When document-unit supervision is requested, the supervised cell is built with `leaf_unit_count = 1` because the gold labels attach to individual document units. Grouped leaves apply to root-only cells.

Leaf-grouping grid examples use `treepo.methods.grid` for cell enumeration and JSON/CSV output.

## Evidence and Local-Law Certificate Example

`run_local_law_certificate.py` is the smallest end-to-end evidence artifact
walkthrough. It builds the evidence objects a real run emits, directly from
sampled rows:

```bash
uv run python examples/methods/run_local_law_certificate.py \
  --output-dir outputs/local_law_certificate_example
```

The output directory contains sampled local-law rows, an audit summary,
preference exports, `evidence.json`, and `certificate.json`. The certificate is
a component-radius ledger for the artifact shape; the toy estimation radius is
illustrative, and a publication run supplies the task's finite-sample bound.

## Family Axis

`treepo.fit(...)` selects one family with the `family` key. Family-specific
choices live in `backend_config`; for example, `family = "neural_operator"`
uses `operator_kind = "fno"` for the FNO route. The built-in families are
`oracle`, `learnable_constant`, `classical_sketch`, `neural_operator`, `fno`,
`llm`, and `dspy`. The LLM/DSPy families are provider-neutral and accept
injected `predict_fn`, `program`, or `dspy_program` hooks. Local-law penalties
are objective terms.
