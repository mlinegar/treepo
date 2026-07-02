# treepo Examples

Examples are small source-tree fixtures. They are not part of the installed
package API.

## Benchmarks

```bash
treepo-bench run classical-sketches \
  --config examples/bench/classical_sketches.yaml \
  --json-out outputs/classical_sketches.json \
  --csv-out outputs/classical_sketches.csv

treepo-bench run markov \
  --config examples/bench/markov.yaml \
  --json-out outputs/markov.json \
  --csv-out outputs/markov.csv
```

The Markov example is package-native. The central Manifesto/RILE methods example is also package-native, but provider-neutral: downstream code supplies the real DSPy/LLM program.

The Manifesto/RILE example includes a document-unit supervision grid. Root-only cells can sweep leaf grouping sizes because they use document labels for `f`; unit-supervised cells add gold `TaskState` labels for `g` with one document unit per leaf. The packaged fixture uses qsentences, while downstream tasks can provide paragraph, section, or extractor-span units through the same shape.
Manifesto cells can also sample training documents and qsentences with known
design propensities; the example writes document and qsentence sampling JSONL
sidecars and stores joint propensities in exported preference units.
`run_manifesto_end_to_end.py` is the complete packaged walkthrough: sampled
docs/qsentences, `treepo.fit(...)`, evidence JSON, and trainer exports.
`run_manifesto_reward_mechanisms.py` additionally exports root-only,
qsentence-only, and combined DPO/reward/GRPO trainer views.

## Methods

Method examples live under `examples/methods/` and run through `treepo.fit()`.
The local-law certificate example is evidence-only: it shows the sampled
C1/C2/C3 row, preference, statistic, evidence, and certificate artifact shape
without training a model.
