# Semantic Forests

## Definition

A **Semantic Forest** is a population of document C-Trees, or declared views
of those trees, governed by one shared declared semantic mechanism. The
mechanism may be learned or explicitly fixed; that status is part of the run
contract. A Semantic Forest is not a tuple of independently fitted
target-specific C-Trees.

Fix an ordered target catalog

```text
J = ((target_name_j, oracle_id_j))_{j=1}^m
```

and the corresponding joint oracle

```text
F*(X) = (f*_1(X), ..., f*_m(X)).
```

`target_name_j` gives coordinate `j` a stable semantic identity.
`oracle_id_j` names the versioned evaluator or labeling process that supplies
that coordinate. The identifiers are not interchangeable: one oracle can
label several targets, and a stable target can survive an oracle revision.

The system has one shared state mechanism `g_hat` and one named vector readout
`F_hat`. For every document `X_i` and any predeclared tree view
`v`, the ordinary C-Tree recursion builds a document tree

```text
T_{i,v} = T(X_i, v; g_hat)
```

and returns one vector `F_hat(s_root)`, in the catalog's canonical order. The
forest is the population `{T_{i,v}}` together with the shared `g_hat`, shared
`F_hat`, target catalog, and view/tree policy. Multiple views may use different
pinned chunkings or topologies, but they do not own different target models.

This distinction is load-bearing:

- Targets are named coordinates of one joint learning problem.
- Documents and views are the members of the forest.
- One declared `g` processes every eligible leaf and internal call; there is
  no separately learned reducer.
- The same joint vector readout returns every declared target.
- Running `fit()` independently `m` times is a product ensemble or multitask
  baseline, not the Semantic Forest object defined here.

## Representation Contract

Topology, whether `g` is invoked, target width, and model family are
independent axes. Every recursive binary C-Tree has `L >= 1` leaves and exactly `M = L - 1`
merge applications. Its base case is a singleton C-Tree: one leaf and no
merges.

The package derives three execution paths:

- `full_doc_direct`: `f(reduce_g(T)) = f(X)`, with one package-owned identity
  call `g(X)=X`;
- `ctree_base_summary`: `f(reduce_g(T)) = f(g(X))`, where a fixed or learned
  nonidentity `g` is called once and no internal call is invoked; and
- `ctree_recursive`: `f(reduce_g(T))` on `L >= 2` leaves, where the same `g`
  is called at leaves and internal nodes.

The reduction is completely induced by that one operator:

```text
reduce_g(Leaf(b))       = g(b)
reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R)).
```

`reduce_g` owns no parameters, artifact, optimizer, or training schedule. Raw
leaf inputs and concatenated child states may require typed packing, but that
packing does not license separate learned functions. Eligible C1 and C3 losses
feed the same training objective and update the one shared `g` artifact used by
the whole fold; there is no separately trained reduction model.
For artifact-aligned comparisons, fit the recursive source once. The base
singleton reuses its exact `(f,g)` pair; the direct singleton reuses its exact
`f` with canonical identity `g`. Both execute zero training iterations.
`model_artifact_contract` carries the exact digests and source/scope IDs across
the result, manifest, evidence, and grid surfaces.


`full_doc` names full-span singleton geometry and may therefore be direct or
summarized. `ctree` names the recursive grammar and includes its singleton
base case. A grid may label its deliberately multi-leaf comparison cell
`ctree`; that narrower experimental arm is `ctree_recursive`, not a new
definition of the grammar.

A deterministic mass-weighted rollup or any other analytic composition rule
is `fixed_analytic`, not learned composition. Reports and evidence must keep
`learned`, `fixed_identity`, and `fixed_analytic` distinct. A full-span
singleton with one nonidentity `g` call remains `full_doc` geometry but has the
`ctree_base_summary` state path. Application-level chunking or internal calls
instead make the path `ctree_recursive`/segmented. The fact that a
family protocol exposes `train_g(...)` does not require every cell to call it.

`g_mode` names the configured policy, while `g_contract.operator` and
`g_contract.learned_this_run` name the observed outcome:

| `g_mode` | Required schedule | Outcome |
|---|---|---|
| `identity` | `f` | `fixed_identity` on the singleton direct path only |
| `fixed` | `f` | an explicitly supplied or family-owned nontrainable operator |
| `learned` with a `g` update | `fg` | `learned_shared` and `learned_this_run=true` |
| `learned` with verified initial artifacts and no update | `fg` | `learned_shared_reused`; the exact prior artifact is evaluated |
| `learned` without an update or verified artifact | `fg` | `trainable_not_updated_this_run`; no learned-`g` evidence |

`train_g_call_count` records attempted calls; `g_update_count` records only
realized updates. Families with in-place state or another artifact-identity
ambiguity return `treepo.methods.GTrainOutcome` to state that outcome
explicitly. A no-op `train_g` call therefore remains
`trainable_not_updated_this_run`. Fixed mode instead requires a concrete,
family-validated nontrainable artifact. The canonical identity artifact is
package-owned. It is restricted to the singleton direct path: binary
composition needs an operator mapping two child states back into one state and
therefore cannot use the unary canonical identity equation.

The last row matters for evaluation-only and shortened ladders: trainability
is a configuration property, not evidence that learning occurred. The fixed
row is the explicit exception to the default skipped-operator interpretation;
its implementation provenance must say what deterministic/frozen `g` was
used. `initial_g_artifact_present` separately discloses a warm start, whose
own artifact provenance may establish learning in an earlier run even though
`learned_this_run=false`.

A realized update in `ctree_base_summary` updates the same shared `g` that a
larger tree would use internally, but the training support contains only the
singleton call. It is not learned-composition evidence; that claim additionally
requires the recursive path, realized internal calls, and suitable C3 evidence.

The generic provider-neutral `llm` family exposes direct scoring and a fixed
text-state/concatenation statistic; it does not optimize `g`. The
optimizer-backed `dspy` family learns joint `f` and one shared `g` program over
role-tagged leaf/merge examples. A learned singleton-summary or recursive
forest still requires an explicit realized update outcome; recursive
composition additionally requires merge-domain support.

All target widths, including `K=1`, `K=3`, and `K=57`, use the same public
`treepo.fit(...)` API, ordered `OracleTargetSpec` catalog, exact named-vector
prediction/target transport, and sum-L1 point metric. Width one is unwrapped
only into compatibility reporting fields; it is not a separate scalar-only
learning path.

The concrete RILE bridge, including compact and expanded component targets
and the comparison design, is specified in `docs/rile_semantic_forest.md`.

## Joint Sufficiency And The C-Tree Laws

The forest uses one task-sufficiency relation for the joint target and its
allowed contexts. Write

```text
s ~^J_epsilon X
```

when state `s` is sufficient for raw object `X` at tolerance `epsilon` for the
declared joint task. This relation is primitive. If `F*` is a fixed
deterministic vector and every coordinate is compared under the same declared
contexts, exact joint equality is equivalent to exact equality of every
coordinate. That algebraic fact does not make an average coordinate loss a
faithful joint diagnostic. Approximate guarantees, partially observed labels,
joint probability laws, and interaction-sensitive downstream decisions may
require joint contextual probes or a structured witness that cannot be
reconstructed from marginal errors alone.

The ordinary C-Tree laws then apply once to the shared `g`:

- C1: `g(x) ~^J_epsilon x` for every allowed leaf.
- C2: if `s ~^J_epsilon X`, then `g(s) ~^J_epsilon X` whenever the grammar
  permits recompression.
- C3: if both child states are jointly sufficient for their raw spans, then
  `g(s_L concat s_R) ~^J_epsilon (X_L concat X_R)`.

For `ctree_base_summary`, the sole nonidentity `g(X)` call gives a realized C1
obligation, but there are no internal nodes and hence no realized C3 rows. The
direct identity path materializes the canonical equation `g(X)=X`. In either
singleton path, the empty C3 stratum is not a pass and does not establish
universal merge closure or `g in G_epsilon`; it says only that root
preservation for this realized tree has no inductive merge step.

Full universal C1/C3, plus C2 when applicable, yields the usual structural
induction for the shared joint relation. It does not follow from low finite
sample loss. A vector readout discrepancy is still normally a readout proxy;
it becomes evidence for the relational laws only when the task supplies a
faithful witness or a calibrated separating probe family.

## One Shared Fit

`OracleTargetSpec` is a coordinate identity record. It carries
`target_name`, `oracle_id`, and optional metadata; it does not own data, a
state kind, an objective, initial artifacts, or a child fit. A single ordinary
`treepo.fit(...)` call owns those shared objects:

```python
import treepo

result = treepo.fit(
    {
        "space_kind": "manifesto_semantic_state.v1",
        "family": "fno",
        "train_data": train_trees,
        "eval_data": eval_trees,
        "initial_artifacts": {"f": None, "g": None},
        "oracle_targets": [
            treepo.OracleTargetSpec(
                target_name="economic_left_right",
                oracle_id="expert_panel_2026_v3",
            ),
            treepo.OracleTargetSpec(
                target_name="immigration",
                oracle_id="expert_panel_2026_v3",
            ),
        ],
        "backend_config": {
            "operator_kind": "fno",
            "target_vector_key": "dimension_scores",
            "node_target_key": "dimension_scores",
        },
        "axis": {"max_iterations": 3},
    }
)
```

For learned operators the grid default `max_iterations=3` is the alternating
`f -> g -> f` sequence; the last `f` is trained after the shared `g` update.
For each fixed target width and family, that sequence runs once on
`ctree_recursive`. The `ctree_base_summary` row then executes iteration zero
with the exact resulting `f` and `g` artifacts. It is a second leaf-count view
of one model, not an independently fitted cell.
The learned DSPy grid uses
`f_record_source="generated_when_available"`, allowing reference states on
the first `f` pass when no learned `g` exists and current-`g` states on the
final pass. `f_record_source="gold_state"` is retained as an explicit
reference-state ablation. Identity/fixed controls elide the inapplicable `g`
slot.

Dense root and node supervision is a mapping keyed by `target_name`, typically
under `dimension_scores`. The ordered `oracle_targets` sequence determines the
only canonical packing order. A mapping with a missing key is not silently
converted to an observed zero.

The result has one `f` artifact, one declared `g` artifact/contract, and one run
manifest. Its `g_mode` records configured trainability; its
`g_contract.operator`, `fit_status`, `learned_this_run`, and update counts
record what happened. A trained shared operator is distinct from
`fixed_identity`, a fixed analytic/family-owned operator, and
`trainable_not_updated_this_run`; an absent training step is not evidence of
learned composition. The run provenance records
`definition="single_shared_g_joint_vector_f_star"`, `target_order`, the
ordered target/oracle records, and a target-catalog digest. The built-in FNO
`f`/`g` artifacts additionally record `training_loss="sum_l1"` and
`training_loss_definition="sum_absolute_coordinate_error"` in their runtime
target schema. That default is the same unnormalized coordinate sum for
`K=1`, `K=3`, and `K=57`; legacy coordinate-mean MSE requires the explicit
`training_loss="coordinate_mean_mse"` opt-in.
Every fit records the width-independent oracle/evaluation metric
`metric="l1"` and
`point_distance="sum_absolute_coordinate_error"`.
Prediction rows expose `target_order`, `oracle_ids_by_target`,
`prediction_by_target`, and `target_by_name`. For more than one target,
`prediction_scalar` is `null` rather than an arbitrary projection of that
vector. The one-target compatibility rule is stated next.

## Unified Vector Implementation And Width-One Compatibility

The public joint-vector fit and metric have no alternate `m = 1` path.
Native vector runtimes
canonicalize task targets as width-`m` tensors and use one model,
normalization, loss, optimizer, and evaluation path at every positive width. A
legacy scalar target `y` is the width-one vector `[y]`; a scalar-only family's
prediction is lifted to `[y_hat]` at the `FamilyRuntime` boundary and projected
back only into the established scalar reporting fields. Such a family
therefore satisfies a named catalog of width one automatically. A catalog
wider than one requires a family that natively learns and returns the declared
joint vector.

An absent target catalog retains the legacy anonymous scalar interface. It is
an unnamed width-one compatibility case, not a zero-target learning problem.
A genuine `m = 0` objective would have no supervised task readout and must be
introduced explicitly rather than overloaded onto legacy scalar fits.

For `m = 1`, dropping only the additive target-name, oracle-provenance, named
prediction mappings, and per-target mirror metrics recovers the legacy scalar
fit. Under the same data, numerical target, initialization, optimizer and
runtime configuration, deterministic backend, and fixed seed, that projection
has:

- the same scalar training targets;
- the same internal neural-operator point loss, because sum-L1 becomes
  ordinary scalar absolute error (or scalar squared error when the explicit
  legacy `coordinate_mean_mse` mode is held fixed);
- the same `g` and `f` parameter updates and final parameters;
- the same predictions after unwrapping the sole coordinate;
- the same legacy scalar metrics; and
- the same dense audit outcome and estimand after the same one-coordinate
  unwrapping.

At the formal level, this transports the scalar oracle/readout, metric, and
sufficiency relation through the scalar-to-singleton-vector isomorphism. It
does not replace contextual task sufficiency with equality of one current
scalar answer.

Concretely, let `iota(y) = (y)` and let `p1` unwrap the sole coordinate. The
singleton lift is

```text
F_star_1 = iota o f_star,        F_hat_1 = iota o f_hat,
p1 o F_star_1 = f_star,          p1 o F_hat_1 = f_hat.
```

Pull back the scalar sufficiency relation and output metric along `iota`, while
holding the state carrier, context family, constructor grammar, masks,
weights, and sampling design fixed. Then C1, C2, full state-state C3, and
`G_epsilon` are the scalar definitions proposition-for-proposition. Likewise,
for the dense coordinate-mean loss,

```text
L_1((y_hat), (y)) = (y_hat - y)^2.
```

The same realized held-out frame therefore gives exactly the same scalar and
singleton HT or AIPW audit estimate sample-by-sample after applying `p1`.
This audit identity does not make the training objective or learned parameters
design-unbiased.

The named run additionally retains its one target/oracle record, catalog
digest, named prediction mappings, and per-target mirror metrics. Existing row
fields, including `teacher_vector`, retain their anonymous-scalar value; the
named truth lives in `target_by_name`. Conservativity therefore does not mean
byte-identical JSON, identical serialized artifact hashes, or identical file
layout. It means equality of the computation and scientific quantities after
forgetting the additive schema.

Anonymous scalar and named width-one checkpoints are computationally compatible
in both directions when their model output widths are one. The resumed artifact
records that compatibility bridge. Named-to-named target/oracle schema drift
still fails, as does a missing-schema checkpoint whose output width exceeds
one; computational width compatibility does not erase declared provenance.

There is no analogous implicit scalarization when `m > 1`. In particular, the
implementation must not silently use the first coordinate, average coordinate
predictions, or pool metrics across targets. Any scalar summary of a joint
vector must be declared as its own readout, loss, probe, or audit estimand.

The package-wide width-one adapter is tested for oracle, classical-sketch,
learnable-constant, provider-neutral LLM, injected scalar,
`neural_operator`, and concrete FNO runtimes. At widths above one, a custom
`family_runtime` is responsible for actually consuming the declared joint
schema; merely advertising target names cannot prove that its internals are
joint-vector faithful.

The target width is the only thing reduced here. Adding document views,
constructor call types, or recompression changes the program grammar and is
not covered by singleton conservativity merely because the oracle width is one.

## Dense Full-Vector MVP

The first implementation deliberately uses **dense block supervision**. At an
observed document root or node, every declared coordinate is present in one
ordered vector; at an unobserved row, none of those oracle coordinates enters
the loss. This all-or-none mask avoids pretending that a missing coordinate is
an observed zero and gives every sampled unit one well-defined joint oracle
bundle.

The default training loss is the same unnormalized sum-L1 point distance:

```text
L_1(y_hat, y) = sum_j |y_hat_j - y_j|.
```

There is no division by target width: at `m = 1` this is exactly ordinary
absolute error. Root targets, supervised node readouts, and vector-state rows
use that one configured reduction. The former coordinate-mean squared-error
surrogate is retained only as
`backend_config["training_loss"]="coordinate_mean_mse"` for reproducing
legacy runs. Optional target normalization fixes the coordinate space before
either reduction and is recorded in the artifact.
Dependence-sensitive alternatives—such as a declared covariance-weighted or
probe-induced distance—should be explicit versioned target metrics. They must
not appear as a hidden `m > 1` code path or be inferred from target count.

This is a useful dense-vector baseline. In `ctree_recursive`, all coordinates
can train one shared `g` and readout across every call in the fold. In
`ctree_base_summary`, they can update that same `g` from singleton calls and
train the readout, but they cannot supply internal-call or C3 evidence. In
`full_doc_direct` and other fixed-`g`
controls, the coordinates train the joint readout while `g` remains fixed. It is
nevertheless a separable loss, and its scientific limitations must remain
visible:

- It weights numerical coordinates uniformly, which is not the same as equal
  scientific importance when target scales differ.
- It contains no explicit interaction term and does not test whether
  cross-target dependencies were preserved.
- Shared parameters can exploit dependencies, but low sum-L1
  does not show that they did.
- A favorable aggregate can hide a poor named coordinate, so per-target metrics
  remain unpooled diagnostics.
- It does not identify arbitrary partial-target masks.

If targets have different units, any fixed normalization belongs in the
declared target/loss schema and must be frozen before evaluation. It must not
be estimated opportunistically from held-out audit labels.

## Dependence-Sensitive Next Step

The next statistical step is a named, versioned joint loss or probe family.
Examples include a fixed positive-semidefinite quadratic loss

```text
L_W = (y_hat - y)' W (y_hat - y)
```

with prespecified off-diagonal interactions, a structured proper scoring rule,
an exact-match or constraint loss, or contextual probes that jointly vary
several semantic coordinates. The loss configuration, target scales, required
coordinate subset, and digest are part of the estimand and artifact schema.

A dependence-sensitive readout loss tests only the dependency encoded by that
loss. It is not automatically a faithful witness for `s ~^J_epsilon X`.
Relational C1/C2/C3 claims require a joint witness or a sufficiently rich,
calibrated separating probe family.

Partial coordinate acquisition comes after the dense path. If a declared loss
decomposes into components `L_h` that require target subsets `S_h`, each
`(call, h)` is its own audit unit. It is observed only when every target in
`S_h` is co-observed, and its inclusion probability is

```text
pi_{u,h} = P(S_h is contained in the observed mask at call u).
```

Marginal target propensities do not identify an interaction term. For example,
an off-diagonal quadratic term for targets `j` and `k` requires the pairwise
co-observation probability `pi_{u,jk}`, not `pi_{u,j} * pi_{u,k}` unless the
design truly makes those draws conditionally independent. An arbitrary
nondecomposable joint loss requires a positive probability of observing its
full required vector. A dense sentinel audit should therefore remain even
after partial-mask training is introduced.

## Held-Out Dense-Block Audit

Sampling can make a declared risk or audit total design-unbiased. It does not
make the learned parameters, representation, or predictions unbiased.

Freeze the following before evaluation:

- target catalog and oracle versions;
- deployed `g_hat` and `F_hat`, including configured `g_mode`, concrete
  operator outcome, `learned_this_run`, and update count;
- document/view and tree policies;
- joint loss or witness/probe definition and target weights;
- any AIPW nuisance prediction.

Generate the eligible held-out frame after those choices are frozen. Keep the
outer document audit and inner constructor-call audit distinct. The root audit
directly measures the deployed vector prediction against a full-document
oracle vector. The node audit diagnoses C1 leaves, C2 recompressions, and C3
merge calls on realized trees. A C3 relational outcome may require both child
antecedent witnesses and the parent consequent, so its sampled unit is the
whole audit bundle rather than merely the parent node.

For fixed rows `u`, fixed target coefficients `rho_u`, joint losses `L_u`, and
known total weight `W = sum_u rho_u`, define

```text
R = (1 / W) sum_u rho_u L_u.
```

Let `O_u` indicate that the complete dense oracle bundle needed for `L_u` was
observed, with actual joint inclusion probability `pi_u > 0`. The
Horvitz--Thompson estimator is

```text
R_hat_HT = (1 / W) sum_u rho_u O_u L_u / pi_u.
```

If documents, calls, and oracle bundles are sampled in stages, `pi_u` is the
actual probability of the complete event. A factorization such as
`pi_doc * pi_call_given_doc * pi_dense_given_call` is valid only when those are
the design's true conditional probabilities. When one selected query returns
the whole vector, the dense-vector propensity is the query propensity, not a
product over coordinates.

With a frozen loss proxy `L_tilde_u`, the loss-level AIPW form is

```text
R_hat_AIPW = (1 / W) sum_u rho_u [
    L_tilde_u + O_u / pi_u * (L_u - L_tilde_u)
].
```

Correct positive propensities make this design-unbiased for the fixed-frame
joint loss for any frozen proxy. A proxy can reduce variance; it does not
repair zero support. The exact claim also requires the full eligible row frame,
a fixed denominator, and no active propensity clipping. A self-normalized
observed-row or Hajek denominator is generally finite-sample biased and must be
labeled as such.

Audit labels used to train `g`, choose targets, select the loss/probes, or fit
the nuisance model cannot also provide an ordinary held-out audit. Use an
independent audit role or cross-fit the entire adaptive pipeline. A persistent
acquisition mask records the actual oracle budget; repeatedly redrawing labels
each training epoch changes that budget and the design.

First-order inclusion probabilities identify the HT point target. Valid
design-based uncertainty under fixed-size or dependent sampling additionally
needs the design's second-order probabilities or valid replicate weights.
Target-coordinate intervals require simultaneous, dependence-aware inference;
the maximum of coordinatewise unbiased estimates is not itself an unbiased
maximum-risk estimate. Node rows and vector coordinates from one document
remain clustered.

## Honest Claims

The dense MVP can support the following statement:

> For a frozen shared compression operator, a prespecified ordered oracle
> vector, and a fixed realized population of eligible document roots or
> constructor calls, the dense-block audit queries the complete oracle vector
> whenever a unit is selected. With positive correctly logged joint inclusion
> probabilities, a fixed target denominator, and no active clipping, the
> Horvitz--Thompson estimator is design-unbiased for the prespecified
> finite-population total or mean of the declared joint vector loss.

That statement is conditional on the realized held-out frame. A deployment
population claim additionally requires a probability sample of documents or a
separate generalization argument. The first implementation must not claim:

- unbiased learned `g`, `F`, parameters, or predictions;
- preservation of cross-target dependencies merely from pooled sum-L1;
- identification under arbitrary partial target masks;
- universal membership in `G_epsilon` from a finite audit;
- relational C1/C2/C3 from a readout loss without a faithful joint witness or
  separating probes;
- a root guarantee from a node mean without the required local-to-root
  violation-count or quantitative transport argument.

## Reporting Contract

One Semantic Forest run should preserve at least:

- the shared state kind, one `f` artifact, one declared `g`
  artifact/contract, its learned/fixed status, the number of merge
  applications, and the tree/view policy;
- ordered `target_order`, `{target_name, oracle_id}` records, and schema digest;
- the named/versioned training and audit loss, scales, and required targets;
- document, tree/view, node/call, law-kind, depth, and audit-bundle identities;
- eligible and observed masks, actual joint propensities, design ID, and role;
- raw named prediction and oracle vectors, plus the precomputed joint loss;
- HT/AIPW totals or fixed-denominator means, support diagnostics, and
  uncertainty method;
- per-target diagnostics kept separate from any explicitly declared joint
  estimand.

There is one result tree, not `m` child-fit result trees. Target-specific
directories or independent fits may be useful comparison baselines, but they
must not be reported as the Semantic Forest realization.
