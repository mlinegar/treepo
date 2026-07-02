---
name: minimal-package-pass
description: Systematic pre-publication simplification pass for a Python package — find and remove dead code, collapse duplication, fix dependency drift, and align docs with code. Use when asked to make a package minimal, coherent, or publication-ready.
---

# Minimal Package Pass

A repeatable procedure for making a Python package minimal, coherent, and
human-understandable. Priority order for every finding: **removal beats
combining beats documenting**; add robustness only where failures are silent.

Load the target repo's anti-pattern catalog first (`docs/antipatterns.md` in
this repo, or the nearest equivalent) and check every audit finding against it
— the catalog carries the smell list, worked examples, and detection recipes
this skill assumes.

## Phase 1 — Baseline

- Require a clean git state with everything committed, so deletions are
  recoverable from history. If the worktree is dirty, stop and ask for a commit.
- Run the full test suite and record exact counts (`N passed, M skipped`,
  runtime). Every later phase diffs against this number.
- Note the packaging gates that exist (release checks, layer tests, export
  pins, `uv lock --check`) — they get updated in the same change as any
  deletion they pin.

## Phase 2 — Map the contract

Deletions are constrained by who consumes the package. Before any audit:

- Enumerate every external consumer: sibling repos, workspaces, deploy
  scripts. For each, collect exact imports:
  `rg -n "from <pkg>|import <pkg>" <consumer>/src <consumer>/scripts`
- Produce the **(module, symbol) must-survive list** — exact pairs, with the
  consumer file for each. Distinguish live imports from already-broken ones
  (a consumer import that fails today constrains nothing).
- Add console scripts (`pyproject.toml [project.scripts]`) and every public
  surface the docs promise.

## Phase 3 — Audit fan-out

Run these audits in parallel (parallel agents when available, one area each).
Each auditor reads files fully and returns findings as REMOVE / MERGE /
SIMPLIFY / DOCS lists with file:line evidence, ranked by impact.

1. **Dead-code audit (per area).** Build the who-imports-whom graph; classify
   every module: dead / test-only / example-only / live.
   Dead module: `rg "from pkg.m import|import pkg.m" src tests examples scripts`
   returns hits only under `tests/` (test-only) or nowhere (dead).
   Within live modules, check each public symbol the same way.
2. **Duplication scan.** Same helper defined in multiple files, sibling files
   under ~20% divergence (`diff` them), repeated guarded-import blocks,
   constants defined more than once.
3. **Speculative-generality scan.** Every config field:
   `rg "field_name" src tests examples` plus TOML/YAML configs — a field that
   appears only in its dataclass is decoration. Also: alias keys nothing
   passes, single-valued Literals, registry hooks never exercised, guards on
   states upstream validators already reject.
4. **Dependency audit, both directions.** Declared-never-imported:
   `rg "import <name>|from <name>" src` for each declared dep.
   Imported-never-declared: every third-party import in src needs a matching
   declaration (watch transitive riders like scipy-via-sklearn and
   version-conditional deps like tomli on older Pythons). Extras must appear
   in the README.
5. **Prose audit.** Negative-space language:
   `rg -n "does not|do not|not part of|belongs outside|remains as|instead of|no longer" -g '*.md' README.md docs examples`
   Stale claims (verify each against the tree), dated working-note docs shipped
   in `docs/`, and doc-vs-code drift in module/family/extras/command lists.

## Phase 4 — Decide

Synthesize the audits into one change plan. Decision rules:

- **Removal is the best simplification.** Delete verified-dead code outright
  from the committed baseline — git history is the archive.
- Combine second: parametrized base over copy-paste siblings, one bottom-layer
  helper over three private copies, fold single-caller modules into their
  caller while keeping public import paths stable.
- Document third, and only what survives.
- Keep theory- or paper-relevant code even when test-only — flag it explicitly
  in the report instead of deleting.
- Anything on the must-survive list is untouchable; verify before changing any
  public signature by reading the consumer's actual call sites.

## Phase 5 — Implement in phases with strict file ownership

- Split implementation so no two concurrent agents share a file. A working
  split: (a) top-level package + core subsystem, (b) remaining subpackages +
  pyproject + inventory, (c) docs/README/examples prose. Shared files
  (`__init__.py`, release gates, layer tests) belong to exactly one phase.
- Code phases run before the docs phase when sequential; when parallel, the
  docs phase writes against the prescribed end state and the orchestrator
  reconciles skips afterward.
- **Every code phase ends green**: full suite, import-laziness check
  (`python -c "import sys, pkg; assert 'torch' not in sys.modules"` or the
  repo's equivalent), an explicit import of every must-survive symbol, and the
  release/packaging gates.
- Update gates, export pins, and inventory entries in the same change as the
  deletion they pin — the tree is never red between phases.
- An item that turns out to have a hidden consumer or to need redesign gets
  skipped and reported, never half-applied.

## Phase 6 — Verify and report

- Full suite vs baseline (explain every count change), release gates, laziness
  check, must-survive import check, and a CLI/example smoke run if the package
  ships one.
- Report: net line delta (`git diff --stat`), per-area summary of what was
  removed/combined/simplified, skipped items with reasons, and any follow-ups
  (e.g. consumer-side imports that were already broken before the pass).

## Style rules for any prose the pass touches

- State what the code does, in present tense — rewrite sentences about what it
  does not do, has not done, or "belongs elsewhere until". Positive laziness
  guarantees are the pattern: "torch loads only when a neural-operator family
  runs."
- Ship reference docs; keep dated working notes (handoffs, completed plans) in
  the session or tracker, and delete them from `docs/`.
- Verify every command, path, and list in a doc against the tree at writing
  time (`ls` the modules, run `--help` on the commands).
