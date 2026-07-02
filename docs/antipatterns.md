# Anti-Pattern Catalog

This is the catalog of code and prose smells that simplification passes on this
package look for, with detection recipes. Each entry names the smell, shows a
concrete example found and removed here, and states the rule to follow. Read it
alongside [`methods_module_layout.md`](methods_module_layout.md) before a
cleanup or review pass.

Throughout, "consumer" means an import or call site outside the defining module
and its own test file — in this repo's `src/`, `tests/`, `examples/`, `scripts/`,
or a registered downstream workspace.

## 1. Facade-only decomposition

**Smell.** A module tree that *looks* decomposed: `_x_core.py` re-exports forty
private symbols "so existing `from pkg._x_core import ...` call sites stay
stable" — and a search finds zero such call sites. The hub's justification is a
claim about consumers that don't exist, and every reader pays the indirection.

**Example.** `_preference_core.py` was a pure re-export hub imported by exactly
one file (its own facade), and `_neural_operator_core.py` carried a 40-symbol
private `__all__` for outside importers that a repo-wide search showed had never
existed. Both re-export layers were deleted; the facades now import directly
from the responsibility modules.

**Rule.** Point facades at responsibility modules directly. Add a re-export hub
at the moment a second import surface actually appears, and name that consumer
in the hub's docstring.

**Detect.** For each `_*_core.py` or hub: `rg "from .*_x_core import|import .*_x_core" -g '!<the hub itself>'` — an empty result means the hub serves nobody.

## 2. Dead contract modules

**Smell.** A formal dataclass contract (`FooContract`, `FooRow`, validators,
digests) that the production writer bypasses. The contract certifies a shape
that nothing produces or reads; only its own test keeps it warm.

**Example.** `manifest.py` defined `RunManifestContract`, `ManifestRow`,
`ArtifactLineage`, and camelCase `from_dict` aliases with zero producers — while
the actual manifest writer, `methods/_run_manifest.py`, built its own ad-hoc
JSON dict. The one live symbol (`stable_digest`) moved to `common.py`; the rest
was deleted.

**Rule.** Make the production writer construct the contract type, or delete the
contract. A contract earns its keep by being on the write path.

**Detect.** For each contract class: `rg "ClassName\("` outside its module and
test — construction sites are the proof of life. Check the writer module
actually imports it.

## 3. Test-only public API

**Smell.** A module or function whose only consumer is its own test file. The
test passes forever and certifies code nobody runs.

**Example.** `honesty.py` (132 lines) was imported solely by
`test_unified_contracts.py`; `tree.load_tree_records` and
`local_law.corrected_losses_from_rows` had the same single-test lifecycle. All
were deleted with their test sections; the JSONL round-trip test now reads the
file directly.

**Rule.** Ship API that a workflow, example, or downstream consumer exercises.
And run the consumer check over EVERY workspace root before classifying: the
uniform-node IPW wrappers in `training/local_law.py` were classified test-only
by a search that covered the main downstream repo's `src/`, `scripts/`, and
`tests/` — and missed four live callers under its `parallel/` workspace. The
removal stood (the wrappers are downstream sampling plumbing over the general
`local_law_objective_from_losses`, now vendored beside their callers), but the
misclassification cost a repair pass. The must-survive list from step 2 of the
removal pass is only as good as the roots it enumerates.

**Detect.** Build the import graph: for module `m`,
`rg "from treepo.m import|import treepo.m" src examples scripts` plus downstream
repos. Hits only under `tests/` classifies it test-only.

## 4. Speculative generality

**Smell.** Configuration surface for futures that never arrived: fields nothing
sets, alias keys nothing passes, a `Literal` with one admissible value guarding
an unreachable `raise`, registry hooks never exercised.

**Example.** `target_keys` was never set anywhere (`target_key` +
`target_vector_key` covered every real use); `use_numeric_leaf_features` was
never disabled; `gepa_kwargs` was plumbed into artifacts but never populated.
All removed; surviving behavior is unconditional. The same audit flagged
`make_hll_adapter`'s `backend` and `hash_bits` arguments — the removal pass
then found live callers (a downstream runtime passes `backend=`; the packaged
HLL example config sets `hash_bits`), so both stayed. Running the detection
recipe before deleting is what separates the two outcomes.

**Rule.** Add a knob together with the first caller that sets it. One concrete
code path beats three configurable ones.

**Detect.** For every config field: `rg "field_name" src tests examples` plus
any TOML/YAML configs. A field that appears only in its dataclass definition is
decoration.

## 5. Unreachable defensive code

**Smell.** Guards against states the program has already excluded: an
`isinstance` check after an unconditional coercion, `hasattr` on an attribute
the class always defines, an `except` clause listing exceptions the callee
can't raise, a fallback arm for input an upstream validator already rejects.

**Example.** `learning._fit_learning` checked `isinstance(spec,
CTreePOLearningSpec)` two lines after `spec = dict(...)` guaranteed a dict,
guarded `hasattr(CTreePOLearningSpec, "from_mapping")` (always true), and
round-tripped a `FitResult` through `to_dict()` into an identical `FitResult`.
`__init__.py` caught `(PackageNotFoundError, TypeError, KeyError)` from a call
that raises only the first. `TreeRecord.root()` kept a third fallback reachable
only for cyclic graphs `validate_tree_record` already rejects.

**Rule.** Let the type system and upstream validators do their job once; write
the straight-line code they license. Catch exactly the exceptions the callee
documents.

**Detect.** Read each `isinstance`/`hasattr`/`except` and ask what call sequence
reaches it with the guarded state. Trace one; when the trace requires an
impossible input, the guard goes.

## 6. Copy-paste siblings

**Smell.** Near-identical files or helpers maintained in parallel, drifting one
bugfix at a time.

**Example.** The four datasketches quantiles adapters were ~95% identical
(differing in sketch constructor, config dict, and one accessor); the guarded
`import datasketches` block was pasted into five files; `_safe_float` existed
three times, `MIN_PROPENSITY = 1e-12` three times, and `jsonable`/`json_default`
in several variants. Each collapsed to one parametrized base or one bottom-layer
helper (`methods/_coerce.py`, `common.py`).

**Rule.** The second copy is the signal: extract a parametrized base class or a
shared bottom-layer helper at copy number two, and import it everywhere.

**Detect.** `rg "def _safe_float|MIN_PROPENSITY ="` style searches for known
helpers; for whole files, compare siblings in the same directory with `diff`
— under ~20% divergence means one parametrized implementation.

## 7. Zero-value indirection

**Smell.** A hop that adds a name without adding behavior: one-line wrapper
functions, two-hop facades, an import-shim module whose "deferral" is undone on
the first call, serialize-then-parse round-trips.

**Example.** `cli.py` existed as an "import-light home" whose `main()`
immediately imported `treepo.bench.cli` anyway — the console script now points
at `treepo.bench.cli:main` directly. `tree_record_from_value` aliased
`TreeRecord.from_value`. `runner.py` computed `json.loads(summary.to_json())`
where `summary.to_dict()` says the same thing once.

**Rule.** Every layer earns its place by transforming, validating, or isolating
something. Route callers to the real symbol and delete the hop.

**Detect.** Grep for one-line `def f(...): return g(...)` bodies and
`json.loads(.*to_json` round-trips; for shim modules, check whether the shim
does anything besides import-and-call.

## 8. Tombstone functions

**Smell.** A function that exists only to raise "this lives elsewhere". It
occupies public surface to describe an absence.

**Example.** `lda.run_lda_recovery` existed solely to raise ImportError
explaining that recovery is application code. Deleted: the family registry's
natural unknown-name error plus one docs sentence carry the same information.

**Rule.** Document extension points in docs; let registries report unknown
names with their normal error. Reserve callables for behavior.

**Detect.** `rg "raise (ImportError|NotImplementedError)" src` and inspect each
hit: a function whose entire body is the raise is a tombstone.

## 9. Workarounds guarding dead code

**Smell.** A carefully engineered — and documented — workaround protecting a
code path that has no callers. The workaround's prominence hides the deadness:
reviewers admire the mechanism and never ask who uses the function.

**Example.** The module-layout doc devoted a paragraph to a sanctioned
function-local-import back-edge inside `PreferenceDataset.filter_tree` — and
`filter_tree` had zero callers anywhere. Deleting the function deleted the
back-edge and the paragraph.

**Rule.** Verify the consumer exists before engineering around a constraint,
and re-verify when documenting it. A workaround's documentation should name who
needs it.

**Detect.** For every documented special case, run the consumer check from §3
on the code it protects.

## 10. Write-only data

**Smell.** State that flows in one direction only: attributes assigned and
never read, payload keys serialized and never loaded, twin fields carrying one
number.

**Example.** `NeuralOperatorFamily._target_mean` was assigned in two places and
read in zero. `PreferenceDataset.to_dict` embedded a full `"records"`
projection that `load()` ignored — doubling every saved dataset file.
`InfluenceWeightedAuditOverlap` carried `W_lambda`/`max_weight` and
`n_rows`/`n_total` as identical pairs.

**Rule.** For every write, name the reader. Serialize exactly what `load()`
reads back.

**Detect.** Per attribute: `rg "\._target_mean"` and split hits into
assignments vs reads. Per payload: diff the keys `to_dict` writes against the
keys `from_value`/`load` consumes.

## 11. Silent failure

**Smell.** `except Exception: pass` around an export or save. The call reports
success, the artifact is missing, and the debugging session starts three steps
downstream.

**Example.** `_preference_io.export_preference_records` swallowed Hugging Face
save failures with a bare pass. It now records
`files["hf_dataset_error"] = str(exc)` so the returned payload carries the
outcome.

**Rule.** Record every handled failure in the returned payload or raise. The
caller decides severity; the callee reports honestly.

**Detect.** `rg -n "except.*:\s*$" -A1 src | rg -B1 "pass$"`.

## 12. Dependency drift

**Smell.** `pyproject.toml` and the import graph disagree in either direction:
declared packages nothing imports, imports satisfied only transitively, extras
no document mentions.

**Example.** `langextract`, `tiktoken`, and `openai` were declared and never
imported (the client is hand-rolled over `requests`). `scipy` was imported
directly but declared nowhere (riding on scikit-learn's transitive closure), and
`tomli` — the 3.10 fallback for `tomllib` — was undeclared while
`requires-python` said `>=3.10`. The `fno`/`embed-train`/`llm-train` extras
appeared in no documentation.

**Rule.** Declare exactly what `src/` imports; import everything you declare;
document every extra in the README. Re-run the check in both directions after
dependency changes.

**Detect.** For each declared dep: `rg "import <name>|from <name>" src`. For
each third-party import in src: confirm a matching declaration. Diff the extras
table in pyproject against the README's install section.

## 13. Negative-space prose

**Smell.** Documentation that describes absences: what the code does NOT do,
what "belongs elsewhere until" some future event, claims that were once true,
references to deleted artifacts, and dated working notes shipped as
documentation.

**Example.** A comment claimed GEPA defaults were "baked into
`DSPyFamilyConfig.gepa_kwargs`" — the field's default was an empty dict.
`protocol.py` described byte-identity as "the stronger property held by the
native adapter" — an adapter that no longer existed. Three dated plan/handoff
docs (every section "Status: completed") shipped in `docs/`.

**Rule.** State what the code does, in present tense, verified against the
tree at writing time. Positive laziness guarantees work: "torch loads only when
a neural-operator family runs." Keep working notes in the session or issue
tracker; ship reference docs.

**Detect.**
`rg -n "does not|do not|not part of|belongs outside|remains as|instead of|no longer" -g '*.md' README.md docs examples`
and review every hit; separately, spot-check each doc's module/family/extras
lists against `ls` output.

## Running a removal pass

The safe order of operations, distilled from this package's own passes:

1. **Green baseline first.** Run the full suite and record exact counts
   (`N passed, M skipped`); every later phase diffs against it.
2. **Pin the external contract.** Enumerate every consumer repo/workspace, `rg`
   its imports of this package, and write down the exact (module, symbol)
   must-survive list before deleting anything. Include console scripts and
   documented public surfaces.
3. **Classify every module** via the who-imports-whom graph: dead, test-only,
   example-only, live. Deletion decisions read off the classification.
4. **Delete outright from a committed baseline.** Git history is the archive;
   the `OLD_` prefix convention is for superseded-but-referenced legacy code,
   and verified-dead code skips it.
5. **Update gates in the same change** — release checks, export pins, layer
   tests, inventory entries — so the tree is never red between commits.
6. **Re-verify**: full suite, release gates, the import-laziness check, and an
   explicit import of every symbol on the external-contract list.
