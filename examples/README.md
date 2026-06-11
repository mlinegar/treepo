# treepo Research Examples

Runnable examples are intentionally kept under `examples/research/` rather than the minimal package surface. They are small fixtures for bench, runtime, and method workflows without copying workspace-scale scripts into the release boundary.

## Runnable Today

### Central Paper Suites

Run every package smoke example through one command:

```bash
treepo-bench suite paper-smoke --out-root outputs/paper_smoke --jobs 1
```

Emit the full paper-grid commands without launching them:

```bash
treepo-bench suite paper-grids \
  --out-root outputs/paper_grids \
  --jobs 1 \
  --commands-only \
  --emit-commands outputs/paper_grids/commands.sh
```

`paper-grids` composes the cardinality, classical-sketch, and LongBench runtime smoke grids under one output root. Use filters such as `--seeds`, `--capacities`, and `--leaf-counts` to shrink the grid.

### Core HLL/Cardinality

Uses the learned-sketch/cardinality recovery benchmark with a tiny CPU config:

```bash
treepo-bench run cardinality-recovery \
  --config examples/research/bench/cardinality_recovery.yaml \
  --json-out outputs/cardinality.json \
  --csv-out outputs/cardinality.csv
```

### HLL Merge Learning

Uses a tiny PyTorch HLL merge-learning config:

```bash
treepo-bench run hll-merge-learning \
  --config examples/research/bench/hll_merge_learning.yaml \
  --json-out outputs/hll_merge_learning.json \
  --csv-out outputs/hll_merge_learning.csv
```

### Classical Sketches

Uses the broad mergeable-sketch comparison surface:

```bash
treepo-bench run classical-sketches \
  --config examples/research/bench/classical_sketches.yaml \
  --json-out outputs/classical_sketches.json \
  --csv-out outputs/classical_sketches.csv
```

## Runtime Examples

These fixtures run through the package-native `longbench-runtime` experiment. They default to deterministic local/mock behavior, so they do not require a live model server.

### LLM Full Context

`runtime_llm_full_context.yaml` uses only a `scorer` endpoint. This corresponds to direct full-context scoring.

```bash
treepo-bench run longbench-runtime \
  --config examples/research/runtime/runtime_llm_full_context.yaml \
  --json-out outputs/runtime_llm_full_context.json \
  --csv-out outputs/runtime_llm_full_context.csv
```

### Embedding Retrieval

`runtime_embedding_retrieval.yaml` adds an `embedder` endpoint and uses retrieval before scorer prediction.

```bash
treepo-bench run longbench-runtime \
  --config examples/research/runtime/runtime_embedding_retrieval.yaml \
  --json-out outputs/runtime_embedding_retrieval.json \
  --csv-out outputs/runtime_embedding_retrieval.csv
```

### Summary Tree

`runtime_summary_tree.yaml` uses the `summarizer` role for leaf/tree evidence and the `scorer` role for the final choice.

```bash
treepo-bench run longbench-runtime \
  --config examples/research/runtime/runtime_summary_tree.yaml \
  --json-out outputs/runtime_summary_tree.json \
  --csv-out outputs/runtime_summary_tree.csv
```

### FNO / State Model

`runtime_fno_state_model.yaml` adds a `state_model` role. The state model selects or renders evidence; a `scorer` still produces the final prediction.

```bash
treepo-bench run longbench-runtime \
  --config examples/research/runtime/runtime_fno_state_model.yaml \
  --json-out outputs/runtime_fno_state_model.json \
  --csv-out outputs/runtime_fno_state_model.csv
```

### All Runtime Methods

`runtime_all_methods.yaml` runs `full_context`, `retrieval`, `summary_tree`, `state_tree`, and `neural_operator` against the tiny fixture.

```bash
treepo-bench run longbench-runtime \
  --config examples/research/runtime/runtime_all_methods.yaml \
  --json-out outputs/runtime_all_methods.json \
  --csv-out outputs/runtime_all_methods.csv
```

## Tiny Data Fixtures

`examples/research/runtime/longbench_v2_tiny.yaml` is a one-row LongBench v2-shaped fixture for documentation and parser tests.
