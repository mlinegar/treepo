# Full-Document Structured-LLM Contract

`treepo` provides a small provider-neutral artifact boundary for full-document
LLM benchmarks. The package owns strict JSON records and task arithmetic. A
downstream application owns private documents, prompt/DSPy optimization,
provider SDK calls, batching, retries, credentials, and locked-split policy.

## Canonical representation semantics

`full_doc` names full-span singleton geometry: one document-sized leaf and zero
tree merges. It does not, by itself, fix `g_mode`. In the recursive C-Tree
grammar this is the `L = 1`, `M = L - 1 = 0` base case.

The structured adapter documented here is specifically the
`full_doc_direct` comparator: it computes the universal equation
`f(reduce_g(T))` on one leaf and materializes the package-owned identity
`g(X)=X`. Its operator status is `fixed_identity` and it must not be reported
as having learned `g`. In an aligned comparison it reuses the exact source
`f` at zero iterations rather than fitting a separate direct model.

A summarized singleton is also a valid full-span C-Tree. Its path is
`ctree_base_summary`, it computes `f(reduce_g(T)) = f(g(X))`, and it may call a fixed or learned
nonidentity `g` once even though it has no internal nodes. A realized update
changes the same shared `g` that a larger tree would use recursively, but the
singleton supplies no internal-call or C3 evidence for learned composition.
The provider-neutral `llm` family does not optimize `g`. The optimizer-backed
`dspy` family can learn this shared nonidentity `g`, using the same program at
leaf and merge calls, and must return an explicit realized update outcome.

There is never a separately learned `reduce_g`. For any larger tree,
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`, using the
same artifact and parameters at every call.

Tokenization inside a model does not change semantic tree shape.
Application-level chunking, partial-document scoring, or an explicit fold
does: that is a segmented/`ctree_recursive` path. A deterministic mass-weighted
or other analytic `g` is `fixed_analytic`, not learned, and cannot be used as
evidence for learned composition.

The same full-document adapter supports `K=1`, `K=3`, and `K=57` through the
ordinary `treepo.fit(...)` API and ordered named target-vector contract. Every
width uses the point distance `sum_j |prediction_j-target_j|` followed by a
document mean. The sole K=1 coordinate may be unwrapped only for legacy scalar
reporting; it is not a different prediction or metric path.

## Structured request and result artifacts

Use `treepo.llm.JSONRequestRecord` to freeze a model name, arbitrary JSON
request payload, and non-secret provenance. Use `JSONResultRecord` to join one
validated JSON object to `TokenUsage`. `TokenUsage.from_value(...)` normalizes
input, cache-read input, cache-write input, output, reasoning, and total tokens
while retaining unknown provider usage fields in `metadata`. Cache-write
aliases such as `cache_write_tokens` and `cache_creation_input_tokens` are
normalized to `cache_write_input_tokens`; conventional cache-read aliases are
normalized to `cached_input_tokens`.

Cache categories are non-negative provider-reported counters, not a portable
partition. Providers differ on whether `input_tokens` includes cache reads and
writes and on whether the two cache categories can overlap. The contract
therefore does not require either counter, or their sum, to be at most
`input_tokens`. A cost ledger must apply the documented semantics of its
specific provider rather than deriving ordinary input by subtraction unless
that partition is guaranteed.

These records do not send requests. In particular, credentials and HTTP
headers are not fields in the contract; credential-like keys are rejected
from persisted record payloads and metadata.

```python
from treepo.llm import JSONRequestRecord, JSONResultRecord
from treepo.tasks.manifesto import manifesto_rile_mass_output_schema

request = JSONRequestRecord(
    request_id="blind-doc-001:model-a:rep-0",
    model="model-a",
    payload={
        "instructions": frozen_rubric_and_training_documentation,
        "document": complete_document_text,
        "output_schema": manifesto_rile_mass_output_schema(),
    },
    metadata={
        "split": "validation",
        "program_digest": frozen_program_digest,
        "document_digest": document_digest,
    },
)

# A downstream transport obtains `provider_output` and `provider_usage`.
result = JSONResultRecord(
    request_id=request.request_id,
    model=request.model,
    output=provider_output,
    usage=provider_usage,
    metadata={"request_digest": request.digest},
)
```

Persist `request.to_dict()` and `result.to_dict()` as JSON/JSONL and retain
their stable `request.digest` / `result.digest` values. Store the
credential only in the downstream transport environment, never in either
record.

## Manifesto full-document RILE mass state

`manifesto_rile_mass_output_schema()` returns a strict JSON Schema object with
four non-negative fields:

- `left_mass`: substantive left-coded policy mass;
- `right_mass`: substantive right-coded policy mass;
- `other_mass`: substantive mass that is neither left nor right;
- `header_mass`: non-substantive/header mass excluded from the denominator.

All fields use one common arbitrary unit. Counts, shares, or another common
scale are valid; the readout is scale invariant. The non-header mass must be
positive. The model does not supply RILE itself. Instead,
`manifesto_rile_mass_state_from_value(...)` validates the output and constructs
the existing public `TaskState(kind="manifesto_policy")` shape. The package
then computes exactly

```text
RILE = 100 * (right_mass - left_mass)
             / (left_mass + right_mass + other_mass)
```

```python
from treepo.tasks.manifesto import (
    manifesto_rile_mass_readout,
    manifesto_rile_mass_state_from_value,
)

state = manifesto_rile_mass_state_from_value(result.output)
prediction = manifesto_rile_mass_readout(state)

# TaskState serialization is a checked round trip.
assert manifesto_rile_mass_state_from_value(state.to_dict()).to_dict() == state.to_dict()
```

This adapter is deliberately `full_doc_direct`-only: it represents one root
judgment and rejects item-level labels. Learning `g` from singleton calls,
segmentation, recursive tree construction, and local-law supervision are
separate benchmark lanes and should not be silently mixed into this
comparator.

For the matched component-vector comparator and its C-Tree Semantic Forest
counterpart, see `docs/rile_semantic_forest.md`. The existing four-mass output
is the compact full-document structured program; the component study freezes
a common normalized target and deterministic readout across its full-document
and tree arms.

## Recommended downstream freeze boundary

Before a locked evaluation split, freeze and digest the complete instruction
rubric, training documentation/examples, output schema, model identifier,
reasoning/sampling configuration, ensemble rule, document normalization, and
request construction code. Keep gold labels and previous predictions outside
the prediction process. Join predictions to gold only after all result records
are immutable.
