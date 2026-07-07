# Preference Data

`PreferenceDataset` is the canonical package boundary for root labels,
node-level labels, scored candidates, pairwise preferences, ranked groups, and
online annotation outputs. It stores two tables: units and candidates.

A unit says what is being supervised. A candidate says what value, summary, or
state was proposed for that unit and how it was judged.

## Shape

Unit fields:

- `unit_id`: stable id such as `doc_a:root` or `doc_a:leaf_0`.
- `unit_type`: `root`, `leaf`, `merge`, `qsentence`, `node`, or another task
  unit name.
- `target`: `f` for root/readout supervision, `g` for state or summary
  supervision, or `both`.
- `context`: prompt text or JSONable context for the supervised unit.
- `tree_id`, `doc_id`, `node_id`, `level`, `position`, `parent_id`,
  `left_child_id`, `right_child_id`: optional tree alignment fields.
- `weight` and `propensity`: design weight and inclusion probability. The
  exported `sample_weight` is `weight / propensity`.
- `metadata`: JSONable annotation, provenance, sampling, or task metadata.

Candidate fields:

- `candidate_id`: stable candidate id.
- `value`: scalar label, text response, JSONable object, or `TaskState`.
- `score`: numeric quality/reward for this candidate.
- `rank`: listwise rank, where `1` is best.
- `preferred`: pairwise/supervised preferred flag.
- `metadata`: JSONable candidate provenance.

For a gold label, put the actual label in `Candidate.value` and mark the
candidate as preferred with a high score. For judged alternatives, keep each
candidate as a row and use `score`, `rank`, or `preferred` to express the
judgment.

## Root-Level Scores

Root-level score supervision trains or evaluates `f`, so use
`unit_type="root"` and `target="f"`.

```python
import treepo
from treepo import Candidate, PreferenceDataset, PreferenceRecord

preferences = PreferenceDataset.from_records([
    PreferenceRecord(
        record_id="doc_a:root:gold",
        unit_id="doc_a:root",
        unit_type="root",
        target="f",
        context="Score document A.",
        tree_id="doc_a",
        doc_id="doc_a",
        node_id="root",
        candidates=(
            Candidate(id="gold", value=0.7, score=1.0, preferred=True),
        ),
    )
])

result = treepo.fit({
    "family": "dspy",
    "train_data": train_trees,
    "eval_data": eval_trees,
    "preference_data": preferences,
    "backend_config": {"predict_fn": predict_fn},
})
```

The same data can be loaded from flat candidate rows:

```python
from treepo import PreferenceDataset

rows = [
    {
        "unit_id": "doc_a:root",
        "unit_type": "root",
        "target": "f",
        "context": "Score document A.",
        "tree_id": "doc_a",
        "doc_id": "doc_a",
        "node_id": "root",
        "candidate_id": "gold",
        "value": 0.7,
        "score": 1.0,
        "preferred": True,
    }
]

preferences = PreferenceDataset.from_flat_rows(rows)
```

## Node-Level Labels

Node-level supervision trains or evaluates `g`, so use `target="g"` and store
the desired local state or summary in `Candidate.value`. Structured states
should use `TaskState`.

```python
from treepo import Candidate, PreferenceDataset, PreferenceRecord, TaskState

gold_state = TaskState(
    kind="policy_signal",
    counts={"positive": 1.0},
    measures={"score": 0.8},
    text="specific positive evidence",
    metadata={"source": "gold_node_label"},
)

preferences = PreferenceDataset.from_records([
    PreferenceRecord(
        record_id="doc_a:leaf_0:gold",
        unit_id="doc_a:leaf_0",
        unit_type="leaf",
        target="g",
        context="Encode leaf 0 as a composable task state.",
        tree_id="doc_a",
        doc_id="doc_a",
        node_id="leaf_0",
        level=0,
        position=0,
        parent_id="root",
        candidates=(
            Candidate(id="gold", value=gold_state, score=1.0, preferred=True),
        ),
    )
])
```

If labels already live on `TreeRecord` nodes, convert them into supervised
preference units:

```python
from treepo.methods.preference import preference_units_from_trees

preferences = preference_units_from_trees(
    tree_records,
    target="g",
    unit_type="node",
)
```

The helper reads each node's supervised value (`state`, then `label`) and emits
a single preferred `gold` candidate for that node. Root nodes are marked
`unit_type="root"` automatically.

## Online Labels And Sampling

Online labelers can append records as annotations arrive. Keep annotation
provenance in `metadata`; keep sampling design information in `propensity` and
`weight`.

```python
preferences.append(
    PreferenceRecord(
        unit_id="doc_a:leaf_0:label_batch_7",
        unit_type="leaf",
        target="g",
        context="Choose the better state for leaf 0.",
        tree_id="doc_a",
        doc_id="doc_a",
        node_id="leaf_0",
        propensity=0.25,
        metadata={
            "annotator_id": "ann_12",
            "label_batch": "batch_7",
            "document_propensity": 0.5,
            "unit_propensity": 0.5,
        },
        candidates=(
            Candidate(id="specific", value=gold_state, score=0.9, preferred=True),
            Candidate(id="generic", value="generic summary", score=0.2),
        ),
    )
)
```

## Save, Load, Export

```python
path = preferences.save("outputs/preferences.json")
preferences = PreferenceDataset.load(path)

supervised_rows = preferences.to_records("supervised")
dpo_rows = preferences.to_records("dpo")
reward_rows = preferences.to_records("reward")
grpo_rows = preferences.to_records("grpo")
hf_dataset = preferences.to_hf_dataset_dict()
```

Use `treepo.methods.preference.export_preference_records(...)` when a run
should write the canonical JSON file, Hugging Face `DatasetDict`, and trainer
views together.

See `examples/methods/run_preference_optimizer_views.py` for the smallest
task-neutral root/node example, and `examples/methods/run_manifesto_end_to_end.py`
for a package fixture with document and qsentence sampling.
