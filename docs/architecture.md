# treepo Architecture

`treepo` is organized by package intent.

## Layers

- `treepo.methods` plus the top-level value modules: the package's center. See below.
- `treepo.bench.sketches`: the sketch adapter protocol and tree reducer. Optional third-party sketch backends load lazily and name the extra to install.
- `treepo.bench`: benchmark runs, result IO, and release checks. `treepo.bench.classical_sketches` is the comparison benchmark that runs adapters from `treepo.bench.sketches` over shared item streams.
- `treepo.llm`: OpenAI-compatible chat/embedding helpers behind `treepo[llm]`.
  vLLM, SGLang, hosted compatible APIs, and compatible local servers use the
  same `/v1` client; direct local runtimes such as Transformers plug in through
  `predict_fn`.
- `treepo.training`: torch local-law tensor helpers layered on `treepo.local_law`; richer trainers register from downstream packages.
- `treepo.forest`: core-light target/oracle identity for one named joint
  vector evaluated through one declared `g` over a population of document
  trees/views. That operator may be learned or explicitly fixed.
- `treepo.tasks`: small task-specific assets, starting with Manifesto/RILE constants and examples.

## The Methods Layer

`treepo.fit(...)` is the single public learning surface. It normalizes a
mapping spec, resolves one family runtime from the registry in
`treepo.methods.families`, runs the alternating f/g loop in
`treepo.methods.runtime`, and assembles a `FitResult` with metrics, artifacts,
history, and a manifest.

When a spec declares `oracle_targets`, the ordered `OracleTargetSpec` records
name the coordinates and oracle provenance of one joint target vector. The
same entrypoint performs one fit over shared train/evaluation trees, one
state space, one joint `f`, and one declared `g`. It does not repeat the fit
once per target. `g` is learned only in a cell that actually trains it.
A recursive binary C-Tree has `L >= 1` leaves and exactly `M = L - 1` merge
applications, so its singleton base case is one leaf and no merges. Identity
`g` is restricted to that direct, no-composition case; a binary fold requires
an explicit fixed or learned `g`. The package derives three explicit paths:

- `full_doc_direct` computes `f(X)` with identity `g` elided;
- `ctree_base_summary` computes `f(g(X))`, invoking `g` once; and
- `ctree_recursive` computes `f(reduce_g(T))` on `L >= 2` leaves, invoking
  that same `g` at leaves and internal nodes.

The recursion introduces no second model:
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`.
There is one `g` artifact, one parameter set, and one `train_g(...)` update
surface. Eligible C1 and C3 training rows contribute to that same update; there
is no `train_reduce_g(...)`. Input packing may distinguish raw leaves from
concatenated child states, but it must not create another learned operator.

`full_doc` names the full-span singleton geometry and can be direct or
summarized; `ctree` names the umbrella recursive grammar and includes its
singleton base case. Grids may deliberately use a multi-leaf `ctree` arm to
keep comparison cells disjoint. A fixed nonidentity/analytic `g` is a named
control; learned mode means an update occurred or a verified learned artifact
was reused. When a nonidentity singleton operator is updated, it has leaf-call
support only and no internal-call or C3 evidence for composition. The resulting
Semantic Forest is the population of document C-Trees (and any declared views)
processed by the declared operator;
see [`semantic_forests.md`](semantic_forests.md).

Target width does not alter this representation contract. `K=1`, `K=3`, and
`K=57` use the same `treepo.fit(...)` call, ordered named-vector target
transport, and document-mean sum-L1 evaluation. Width one is projected to
legacy scalar reporting fields only after the named-vector computation.

Seven families are built in: `oracle`, `learnable_constant`,
`classical_sketch`, `neural_operator`, `fno`, `llm`, and `dspy`. Downstream
packages register additional runtimes with
`treepo.methods.families.register_family(...)` against the `FamilyRuntime`
protocol in `treepo.methods.contracts`. Wrappers own the training loop and
call `module.train()` / `module.eval()` internally, so every family presents
the same `train_f` / `train_g` / `score_roots_with_f` surface to the runtime.

That common method surface is not common model capability. The generic
provider-neutral `llm` family provides inference/artifact plumbing and a fixed
text-state/concatenation statistic; it does not optimize `g`. The `dspy` family
is optimizer-backed: it learns the joint readout `f` and, for learned C-Tree
paths, compiles one shared `g` over the available leaf/merge call domains.
Artifacts are called learned only after a realized update.

The alternating runtime uses inclusive iteration indices: iteration 0 is the
initial evaluation, odd positive iterations train `f`, and even positive
iterations train `g`. The Semantic-Forest comparison grid requests
`max_iterations=3`, giving `f -> g -> f`; its learned DSPy cells default
`f_record_source="generated_when_available"`, so the first `f` pass can use
reference states before `g` exists and the final `f` pass uses states generated
by the current shared `g`. Setting `f_record_source="gold_state"` remains an
explicit ablation.

DSPy task semantics are injected through `f_signature_instructions` and
`g_signature_instructions`; the Manifesto task owns strict named-vector
instructions for K=1, K=3, and K=57. Warm-start artifacts bind both the ordered
target/oracle catalog and the signature contract. A mismatch fails before
compile/load. Exact package-created `dspy.Predict` programs use state-only JSON
persistence; loading a whole-program pickle is opt-in with
`allow_pickle_program_load=True` and is appropriate only for trusted artifacts.

The optional DSPy token-budget guard is enabled only when all four of
`leaf_size_tokens`, `lm_context_window_tokens`, `max_completion_tokens`, and
`prompt_template_overhead_tokens` are supplied. It checks each training row and
direct live program input. DSPy's compiled demonstration stack is opaque to
`treepo`, so the overhead reservation must conservatively cover it; the
artifact deliberately does not claim an absolute prompt-fit theorem.

Sampling propensities are resolved once through `treepo.sampling` and carried
with source provenance. Preference projections preserve both base and effective
weights so conversion into DSPy examples does not divide twice. This is
optimizer-example weighting, not a Horvitz--Thompson or Hajek estimator, and it
does not imply unbiased learned parameters or predictions.

`treepo.methods.preference` holds the unit-level supervision boundary:
`Candidate`, `PreferenceRecord`, and `PreferenceDataset`, with one canonical
Hugging Face `DatasetDict` shape and generic/supervised/DPO/reward/GRPO
projection exports. See [`preference_data.md`](preference_data.md) for
root-level and node-level loading patterns.

The top-level value modules carry the package's data shapes and diagnostics:

- `treepo.state` — `TaskState`, the JSONable state produced by `g` and read by `f`.
- `treepo.forest` — `OracleTargetSpec`, the coordinate/oracle-provenance
  record for a shared named vector, plus forest-schema identity.
- `treepo.tree` — `TreeNode` / `TreeRecord`, the minimal labeled tree artifact.
- `treepo.statistic` — the executable `ComposableStatistic` protocol for encode/merge/readout.
- `treepo.local_law` — canonical, Lean-aligned scalar C1/C2/C3 row arithmetic and audit summaries.
- `treepo.evidence` — the unified per-run evidence artifact (see [`docs/evidence.md`](evidence.md)).
- `treepo.certificate` — the component-radius certificate ledger.
- `treepo.objective` — objective metadata for manifests and evidence.
- `treepo.sampling` — design-propensity sampling helpers.
- `treepo.artifacts` — canonical run-artifact bundles.
- `treepo.finetune` — trainer-neutral embedding and LLM fine-tuning export views.
- `treepo.common` — small shared utilities such as `stable_digest`.

## Role Vocabulary

Public role metadata follows the paper language:

- `scorer`: practical task scorer `f`
- `summarizer`: learned or explicitly fixed state mechanism `g`
- `oracle`: trusted target/evaluator `f*` or benchmark labels
- `embedder`: vector evidence mechanism
- `state_model`: learned or deterministic state realization

Internal method surfaces are implementation details — chat, embedding, or
operator endpoints all map onto the same public roles.

## Package Inventory

`inventory.yaml` records the package boundary:

- `package`: importable implementation module
- `cli`: public `treepo-bench` command
- `shim`: thin package shim
- `outside`: code owned by downstream packages
- `extension`: optional family or backend registered by another package

Release checks:

```bash
treepo-bench check inventory --json
treepo-bench check hygiene --json
treepo-bench check release --json
```
