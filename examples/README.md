# treepo Examples

Examples are small source-tree fixtures that run from a repo checkout. Each
`examples/methods/run_*.py` file is a self-contained walkthrough: it uses
package fixtures plus local helpers in `examples/methods/example_setup/`, and
when needed loads the paired TOML config in the same directory.

The examples are intentionally wired through the public package records:
`TreeRecord`, `TaskState`, `PreferenceDataset`, `LocalLawAuditRow`, and
`FitResult`. See [`../docs/preference_data.md`](../docs/preference_data.md)
for the canonical root-level and node-level supervision shapes.

## Benchmarks

```bash
uv run treepo-bench run classical-sketches \
  --config examples/bench/classical_sketches.yaml \
  --json-out outputs/classical_sketches.json \
  --csv-out outputs/classical_sketches.csv

uv run treepo-bench run markov \
  --config examples/bench/markov.yaml \
  --json-out outputs/markov.json \
  --csv-out outputs/markov.csv
```

The Markov benchmark is fully local. The LLM methods examples are
provider-neutral: they run with packaged fixtures, and provider-specific code
can be supplied through the same endpoint/callable/program hooks used by
`treepo.fit(...)`. OpenAI-compatible servers such as vLLM and SGLang use
`api_base`; direct local runtimes such as Hugging Face Transformers use
`predict_fn`; DSPy uses an injected program.

The Manifesto/RILE example includes a document-unit supervision grid.
Root-only cells can sweep the leaf-grouping grid because they use document
labels for `f`; unit-supervised cells add gold `TaskState` labels for `g` with
one document unit per leaf. The packaged fixture uses qsentences; another task
can provide paragraph, section, or extractor-span units through the same shape.
Manifesto cells can also sample training documents and qsentences with known
design propensities; the example writes document and qsentence sampling JSONL
sidecars and stores joint propensities in exported preference units.
`run_manifesto_end_to_end.py` is the complete walkthrough: sampled
docs/qsentences, `treepo.fit(...)`, evidence JSON, and trainer exports.
`run_manifesto_reward_mechanisms.py` additionally exports root-only,
qsentence-only, and combined DPO/reward/GRPO trainer views.

## Methods

Method examples live under `examples/methods/` and run through `treepo.fit()`.
The local-law certificate example is evidence-only: it builds the sampled
C1/C2/C3 row, preference, statistic, evidence, and certificate artifacts
directly from sampled rows.

Run any method example with:

```bash
uv run python examples/methods/run_NAME.py --output-dir outputs/NAME
```

For the LLM backend adapter shapes:

```bash
uv run python examples/methods/run_llm_backends.py \
  --output-dir outputs/llm_backends_example
```

Verify the runnable examples and their exported artifacts with:

```bash
uv run pytest -q tests/test_examples.py tests/methods/test_examples_smoke.py
```
