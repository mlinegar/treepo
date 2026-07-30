# RILE as a Semantic Forest

## The proof-of-concept

RILE is a useful bridge from a scalar C-Tree to a Semantic Forest because the
published scalar has a known component decomposition. The Semantic Forest
version does not fit one tree per component. It fits one model over a
population of document C-Trees:

```text
one shared declared operator g
one joint named vector readout F = (f_1, ..., f_K)
one frozen deterministic RILE readout A(F)
```

Independent component fits are retained as a deliberately inefficient product
baseline. They are not the Semantic Forest.

## Representation contract

The C-Tree grammar and execution paths have one meaning across `K=1`, `K=3`,
and `K=57`. A binary C-Tree has `L >= 1` leaves and exactly `M = L - 1`
merge applications:

- `full_doc_direct` is singleton `F(X)` with identity `g` elided;
- `ctree_base_summary` is singleton `F(g(X))`, with one call to a fixed or
  learned nonidentity `g` and no internal calls; and
- `ctree_recursive` is `F(reduce_g(T))` with `L >= 2`, where the same `g` is
  called at leaves and internal nodes.

Here `reduce_g` is only the fold induced by `g`:
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`. There is one
`g` artifact and one training surface, not distinct leaf and reduction models.

`full_doc` names full-span singleton geometry and can be direct or summarized.
`ctree` is the umbrella recursive grammar and includes its singleton base
case. For paired grids, the short `full_doc` arm may be defined specifically
as `full_doc_direct` and the short `ctree` arm as deliberately multi-leaf
`ctree_recursive`; those are disjoint cell choices, not exclusive definitions.
A mass-weighted analytic rollup is `fixed_analytic` and must not be reported as
learned composition. A summarized-singleton update changes the shared `g`
from singleton-call support only; it does not supply internal-call or C3
evidence. A full-span method that chunks or applies the fold internally has the
recursive/segmented path even though its source coverage is still complete.
Target width changes none of this: every width uses the same `treepo.fit(...)`
API, ordered named-vector target contract, and document-level sum-L1
evaluation. The sole `K=1` coordinate is projected to a scalar only for
compatibility reporting.

## Exact targets

Let `N` be the number (or common mass) of non-header quasi-sentences. Let `L`,
`R`, and `O` denote left-coded, right-coded, and all other non-header mass.
Then

```text
L + O + R = N
RILE = 100 (R - L) / N.
```

The middle bucket is called **other**, not neutral. It contains everything
that enters the denominator but is neither in the fixed RILE-left catalog nor
the fixed RILE-right catalog. Such text is not necessarily ideologically
neutral. Headers are excluded from `N`; header mass can be reported as a
separate data-quality diagnostic but is not required by the task readout.

The package exposes two factorizations.

### Compact polarity target (`K=3`)

```text
W_polarity = (L/N, O/N, R/N)
RILE(W_polarity) = 100 (W_right - W_left).
```

This establishes the vector path with the smallest exact target. It also
matches the intuitive left/other/right decomposition.

### Expanded CMP target (`K=57`)

The deliberately expanded target contains the normalized share of each of the
56 canonical CMP policy categories plus one residual non-policy share:

```text
W_cmp = (p_101, ..., p_706, p_residual).
```

The residual carries non-header denominator mass not in the 56 policy
categories, including `000`. Therefore the vector sums to one and identifies
the same RILE exactly. With fixed code sets `C_L` and `C_R`,

```text
RILE(W_cmp) =
    100 [sum_{c in C_R} p_c - sum_{c in C_L} p_c].
```

This is intentionally inefficient for the downstream scalar: most
coordinates have readout coefficient zero. That inefficiency is useful. It
tests whether one shared state preserves a substantially richer semantic
object than the scalar answer alone.

Shares are the task readout, but additive counts are the clean compositional
witness. A law-bearing run may store the corresponding count vector in node
metadata and supervise it through `law_state_target_key`; a weighted leaf-share
rollup uses exact non-header mass as `rollup_weight_key`.

## The comparison

The clean design is a small factorial. It keeps direct readout, leaf
summarization, recursive composition, supervision, and target width separate.

| Arm | Input | Learned target | Supervision | What it identifies |
|---|---|---|---|---|
| `FD-Y` | `full_doc_direct`: `f(X)` | scalar RILE | published root labels | direct full-document readout baseline |
| `SD-Y` | `ctree_base_summary`: `f(g(X))`, no internal calls | scalar RILE | the same observed root labels | effect of one `g` call without recursive composition |
| `CT-Y-root` | multi-leaf `ctree_recursive` | scalar RILE | the same observed root labels | incremental recursive-composition effect |
| `CT-Y-local` | multi-leaf `ctree_recursive` | scalar RILE | separately budgeted root and local labels | value of local supervision |
| `FD-W` | `full_doc_direct` | component vector | component bundle | target-factorization effect with `f` only |
| `SD-W` | `ctree_base_summary` | one joint component vector | the same component bundle design | joint effect of one `g` call without internal calls |
| `SF-W` | multi-leaf `ctree_recursive` | one joint component vector | the same component bundle design | Semantic Forest composition arm |

Three diagnostics complete the picture:

- `Product-W`: one separate scalar fit per component. This tests whether
  sharing `g` and learning the vector jointly helps, but is not a Semantic
  Forest.
- `Gold-W`: exact components followed by the frozen readout. This is the
  arithmetic/data ceiling.
- `Analytic-rollup-W`: learned leaf components with exact mass-weighted
  aggregation. This is a `fixed_analytic` additive control and must never be
  reported as learned `g`; `SF-W` is a learned-composition result only when the
  one updated `g` is exercised recursively and the run carries appropriate
  internal-call/C3 evidence.

Run the comparison as a progression:

1. Compare `FD-Y` with `SD-Y` on identical documents, scalar labels,
   model-selection rules, and label budgets to isolate the effect of one `g`
   call.
2. Compare `SD-Y` with `CT-Y-root` to isolate additional recursive calls of
   that same `g`.
3. Add `CT-Y-local` as a separately named supervision intervention.
4. Run the compact `K=3` `FD-W`/`SD-W`/`SF-W` progression.
5. Run the expanded `K=57` `FD-W`/`SD-W`/`SF-W` progression as the Semantic
   Forest proof-of-concept.
6. Add `Product-W` only as the intentionally inefficient no-sharing
   diagnostic.

`FD-W` is essential: without it, a difference could be caused entirely by the
richer target. `SD-W` is also essential for separating the effect of one
full-document `g` call from the additional effect of recursively applying that
same operator.

The primary downstream gold should remain the published document RILE label.
Also report the exact CMP-recomputed RILE as a mechanism target. Those values
are close but need not be identical; their discrepancy must not be silently
erased.

## L1 at every width

The oracle/evaluation point distance is

```text
d_1(y_hat, y) = sum_j |y_hat_j - y_j|.
```

At `K=1` this is ordinary absolute error. There is no scalar/vector metric
branch. Runs record this contract as `treepo.oracle_metric.v1`, and
`results.json` reports the mean joint L1 next to unpooled coordinate metrics.

The frozen RILE readout gives a useful deterministic bridge. For either share
vector, let `a_j` be `-1`, `0`, or `+1` according to the coordinate's RILE
polarity. Then

```text
|RILE(W_hat) - RILE(W)| / 200
    = (1/2) |a' (W_hat - W)|
    <= (1/2) ||W_hat - W||_1.
```

Thus component L1 controls normalized scalar RILE error. The converse is
false: left and right component errors can cancel and produce perfect scalar
RILE. The package report exposes both values and the bound.

The built-in neural-operator/FNO optimizer uses this same unnormalized
coordinate sum as its default root, node-readout, and vector-state loss:
`backend_config["training_loss"]="sum_l1"`. Thus `K=1` reduces exactly to
absolute error and `K=3`/`K=57` do not divide by target width. The historical
coordinate-mean squared-error surrogate remains available only through the
explicit `"coordinate_mean_mse"` compatibility setting. Artifacts record the
executed choice and definition. RILE task fragments set
`normalize_targets=false`, so this loss is on the declared share coordinates.

Primary metrics:

- raw RILE MAE against published RILE;
- normalized scalar L1, `|error| / 200`;
- mean joint component L1;
- paired per-document MAE differences between arms.

Secondary diagnostics:

- unpooled per-coordinate MAE and calibration;
- component-vector sum, negative mass, and simplex violations;
- exact-CMP RILE error and published-RILE error;
- bias, RMSE, Pearson, and Spearman as non-primary summaries;
- performance by document length, tree depth, and denominator size;
- error versus label/oracle cost.

Do not pool coordinates for Pearson correlation, and do not judge the expanded
target only by aggregate RILE: cancellation is one of the main failure modes
the experiment is designed to reveal.

## Label budgets and leakage controls

One authoritative CMP category annotation yields the whole one-hot component
bundle. Therefore 57 derived coordinates are not automatically 57 independent
human labels. Report at least:

- oracle bundle queries;
- atomic CMP annotations;
- returned coordinate values;
- derived node rows;
- unique source-token exposure;
- API tokens/dollars or human minutes.

For the intentionally inefficient interpretation, also run a lane in which
component oracles are queried separately and charge all 57 queries. Keep that
distinct from the bundled-CMP lane.

Scalar arms must not receive CMP counts, component vectors, or exact
denominator mass unless they are explicitly labeled as known-mass controls.
Component-only arms should omit scalar RILE training labels. The packaged fit
fragment sets `node_target_exclusive=True`, so missing component vectors cannot
fall back to scalar node labels. Freeze splits, code catalog, denominator
policy, target order, readout coefficients, tree topology, and audit masks
before evaluation.

Sampling can make a frozen held-out loss estimate design-unbiased when its
joint inclusion probabilities are known. It does not make learned `g`,
learned `F`, parameters, or predictions unbiased. C1/C3 claims additionally
need a faithful state witness or separating probes; finite component readout
loss alone is not universal membership in `G_epsilon`.

## Package wiring

Authoritative labeled bundles already preserve `cmp_counts` and
`total_non_header_qsentences` at each node. The task adapter densifies those
counts into a fixed target order and attaches the same mapping at the root and
nodes:

```python
import treepo
from treepo.tasks.manifesto import (
    attach_manifesto_rile_components,
    manifesto_rile_component_fit_fragment,
    manifesto_rile_component_output_schema,
    manifesto_rile_component_report,
)

# FD-W uses this exact target order in its structured full-document response.
full_document_schema = manifesto_rile_component_output_schema("cmp56")

# SF-W hydrates the same target from authoritative node CMP counts.
train = attach_manifesto_rile_components(
    train_trees,
    granularity="cmp56",
    drop_scalar_root_label=True,
)
evaluation = attach_manifesto_rile_components(
    eval_trees,
    granularity="cmp56",
    drop_scalar_root_label=True,
)
fragment = manifesto_rile_component_fit_fragment("cmp56")

result = treepo.fit(
    {
        **fragment,
        "family": "fno",
        "schedule": "fg",
        "train_data": train,
        "eval_data": evaluation,
        "backend_config": {
            **fragment["backend_config"],
            "root_weight": 1.0,
            "leaf_weight": 1.0,
            "merge_weight": 1.0,
        },
        "axis": {"max_iterations": 3},
    }
)
```

This call produces one `f` artifact and one `g` artifact. Prediction rows keep
the full named vector.

With learned `g`, three alternating iterations mean `f -> g -> f`, so the
reported readout is refit after the shared operator changes. The paired
learned-DSPy grid defaults
`f_record_source="generated_when_available"`: its first `f` update may use
reference states because there is no learned `g` yet; its final `f` update
uses states generated by the current shared `g`. Use
`f_record_source="gold_state"` only as the explicit reference-state ablation.
Identity and fixed analytic `g` controls run their effective `f`-only
schedule rather than fabricating a `g` update.

Derived RILE is computed explicitly after prediction:

```python
report = manifesto_rile_component_report(
    row["prediction_by_target"],
    row["target_by_name"],
    granularity="cmp56",
)
```

There is deliberately no generic implicit scalarization for `K>1`.

## Paired family-grid example

The package example exposes one grid contract through two family adapters:

```python
from examples.methods.run_manifesto_semantic_forest_grid import (
    run_complete_grid,
    run_family_grid,
)

dspy_backend = {
    "optimizer": "bootstrap_random_search",
    "lm_config": {
        "model": "your-served-model",
        "api_base": "http://localhost:8000/v1",
    },
}
dspy_grid = run_family_grid(
    "dspy",
    "outputs/rile_dspy_grid",
    dspy_execution="learned",
    dspy_backend_config=dspy_backend,
)
fno_grid = run_family_grid("fno", "outputs/rile_fno_grid")
both = run_complete_grid(
    "outputs/rile_family_grids",
    dspy_execution="learned",
    dspy_backend_config=dspy_backend,
)
```

Each family grid has exactly the same nine cells (18 across both families):

```text
K in {1, 3, 57}
  x representation_path in {full_doc_direct, ctree_base_summary, ctree_recursive}.
```

K changes only the ordered `OracleTargetSpec` catalog and output width. The
three paths separately identify direct readout, one singleton `g` call, and
recursive application of that same `g`. They use the same
documents, target construction, reporting schema, and one `treepo.fit(...)`
call per cell; topology changes through ordinary axis/config values rather than
a separate API.

Family changes only the DSPy-versus-FNO backend adapter. Target construction,
exact-key parsing, document roster, sum-L1 point distance, raw-RILE projection,
prediction coverage checks, and package `results.json` semantics are shared.
Every cell keeps common named-vector metrics and exact coverage separate from
optional family-specific telemetry. A downstream comparison-row adapter may
add common compute, acquisition, and measurement-status sections, but it must
not relabel a fixed identity or analytic operator as learned. K=1 remains a
named one-coordinate vector during fitting; its scalar RILE value is projected
only afterward for compatibility reporting.

Run the optimizer-backed DSPy/FNO fixture grid with a JSON or TOML DSPy
backend config:

```bash
uv run python examples/methods/run_manifesto_semantic_forest_grid.py \
  --family both \
  --dspy-execution learned \
  --dspy-config path/to/dspy_backend.toml \
  --output-dir outputs/manifesto_semantic_forest_grid
```

The DSPy C-Tree cells set `g_mode="learned"`; base-summary and recursive cells
use the same shared-g implementation and learner ID. The compiler receives
leaf examples in both cells and merge examples in the recursive cell. These
tiny synthetic records have no teacher-authored summary states, so the example
explicitly sets `allow_identity_g_targets=true` and records
`g_target_source="explicit_reference_text_fixture"`; the learner may not infer
raw-text targets merely because summary labels are missing. Planned
`expected_*` fields are not realized evidence: the executed result must report
the actual `g_update_count`, role support, and shared-g status.

For the deterministic package-interface smoke, select it explicitly:

```bash
uv run python examples/methods/run_manifesto_semantic_forest_grid.py \
  --family both \
  --dspy-execution offline_fixture \
  --output-dir outputs/manifesto_semantic_forest_grid_offline
```

Only `offline_fixture` reads authoritative fixture targets through the
Python program, uses fixed analytic mass-weighted `g`, and sets
`optimizer="none"`. It makes no live LLM calls, executes no DSPy compiler, and
must never be called the learned or empirical DSPy grid. The packaged data are
synthetic in both modes, so the example itself is not model-quality, Polmeth,
or publication evidence.
