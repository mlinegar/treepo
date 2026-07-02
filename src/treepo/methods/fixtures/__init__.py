"""Lightweight fixture builders for method tests and examples.

The package fixtures are small deterministic tree-shaped datasets for HLL,
synthetic LDA, and Markov method checks. This package preserves the historical
``treepo.methods.fixtures`` import surface while keeping each fixture family in
its own module.
"""

from treepo.methods.fixtures.hll import HLLItemTree, make_hll_item_trees
from treepo.methods.fixtures.lda import LDATopicTree, make_lda_topic_trees
from treepo.methods.fixtures.markov import (
    MarkovChangepointTree,
    make_markov_changepoint_trees,
)

__all__ = [
    "HLLItemTree",
    "LDATopicTree",
    "MarkovChangepointTree",
    "make_hll_item_trees",
    "make_lda_topic_trees",
    "make_markov_changepoint_trees",
]
