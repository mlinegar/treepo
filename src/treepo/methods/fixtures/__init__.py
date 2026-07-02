"""Lightweight fixture builders for method tests and examples.

The package fixtures are small deterministic tree-shaped datasets for HLL,
synthetic LDA, and Markov method checks. This package preserves the historical
``treepo.methods.fixtures`` import surface while keeping each fixture family in
its own module.
"""

from treepo.methods.fixtures.hll import (
    HLLItemTree,
    hll_tree_records,
    make_hll_item_trees,
)
from treepo.methods.fixtures.lda import (
    LDATopicTree,
    lda_tree_records,
    make_lda_topic_trees,
)
from treepo.methods.fixtures.markov import (
    MarkovChangepointTree,
    make_markov_changepoint_trees,
    markov_tree_records,
)

__all__ = [
    "HLLItemTree",
    "LDATopicTree",
    "MarkovChangepointTree",
    "hll_tree_records",
    "lda_tree_records",
    "make_hll_item_trees",
    "make_lda_topic_trees",
    "make_markov_changepoint_trees",
    "markov_tree_records",
]
