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

## Methods

Method examples live under `examples/methods/` and run through `treepo.fit()`
or `treepo.methods.fit()`.
