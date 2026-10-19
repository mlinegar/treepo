"""Quickstart: audit a lossy tree summarizer by sampling nodes.

Runs offline in about a second with only the core install (numpy):

    uv run python examples/quickstart.py

Each toy document is split into 8 leaves. A summarizer ``g`` keeps at most
four sentiment words, summaries are merged pairwise up a balanced tree, and a
trusted judge ``f*`` scores net sentiment. Every node gets local-law checks:

* C1 leaf sufficiency: judging a leaf's summary matches judging the leaf.
* C2 idempotence: re-summarizing a summary leaves its judgment unchanged.
* C3 merge consistency: judging a merged summary matches judging its children.

A cheap proxy flags every node where ``g`` had to drop words. Trusted labels
are drawn for a random 20% of checks, and ``audit_local_laws`` combines proxy
and labels into design-based estimates of each law's violation rate.
"""

import random

from treepo.local_law import LocalLawAuditRow, audit_local_laws
from treepo.sampling import apply_node_audit, sample_node_audit
from treepo.schedule import fold_with_trace, merge_children

SENTIMENT = {"good": 1, "great": 1, "win": 1, "bad": -1, "poor": -1, "loss": -1}
BUDGET = 4  # the summarizer keeps at most this many sentiment words


def judge(text):  # f*: the trusted judge (here: net sentiment)
    return sum(SENTIMENT.get(word, 0) for word in text.split())


def summarize(text):  # g: keep sentiment words, up to the budget
    return " ".join([w for w in text.split() if w in SENTIMENT][:BUDGET])


def merge(left, right):  # pairwise merge: re-summarize the two child summaries
    return summarize(left + " " + right)


def proxy(text):  # cheap check: 1.0 if g had to drop anything
    return float(sum(w in SENTIMENT for w in text.split()) > BUDGET)


rng = random.Random(0)
vocab = list(SENTIMENT) + "the a report said vote party plan today".split()
docs = [[" ".join(rng.choices(vocab, k=8)) for _ in range(8)] for _ in range(100)]

rows = []
for d, leaves in enumerate(docs):
    nodes = fold_with_trace([summarize(t) for t in leaves], merge)  # 8 leaves, 7 merges
    children = merge_children(len(leaves))
    for i, summary in enumerate(nodes):
        if i < len(leaves):  # C1: compare with the raw leaf text
            law, source = "c1", leaves[i]
        else:  # C3: compare with the two child summaries
            law, source = "c3", " ".join(nodes[c] for c in children[i])
        rows.append(LocalLawAuditRow(
            row_id=f"{d}:{i}:{law}", law_kind=law, proxy_loss=proxy(source),
            oracle_loss=float(judge(summary) != judge(source)), observed=True))
        rows.append(LocalLawAuditRow(  # C2 on every node
            row_id=f"{d}:{i}:c2", law_kind="c2", proxy_loss=0.0,
            oracle_loss=float(judge(summarize(summary)) != judge(summary)), observed=True))

truth = audit_local_laws(rows)  # toy only: every check has a trusted label
design = sample_node_audit(len(rows), policy="fraction", fraction=0.2, seed=0)
audit = audit_local_laws(apply_node_audit(rows, design))  # trusted labels on 20%

print(f"{'law':22} {'true':>6} {'proxy':>6} {'audit':>6}  labeled")
for law, result in audit["by_law_kind"].items():
    stats = result["local_law_objective"]
    true_rate = truth["by_law_kind"][law]["local_law_objective"]["objective"]
    proxy_rate = sum(r.proxy_loss for r in rows if r.law_kind == law) / stats["row_count"]
    print(f"{law:22} {true_rate:6.3f} {proxy_rate:6.3f} {stats['objective']:6.3f}  "
          f"{stats['observed_count']}/{stats['row_count']}")
