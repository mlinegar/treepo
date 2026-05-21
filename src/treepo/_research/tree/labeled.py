"""
Labeled Tree Data Structures for OPS Training.

This module provides data structures for storing labeled hierarchical trees
of document chunks. Labels can come from any source (ground truth data,
oracle models, LLMs, human annotation) and are treated uniformly.

These trees enable training and testing all three OPS laws:
- Sufficiency: Each chunk has a score (label)
- Idempotence: Re-summarize and compare to original label
- Merge: Parent nodes have labels for merged content

Usage:
    from treepo._research.tree.labeled import (
        LabeledNode,
        LabeledTree,
        LabeledDataset,
    )

    # Create a labeled tree
    tree = LabeledTree(
        doc_id="doc_123",
        document_text="Full document text...",
        document_score=15.0,
    )

    # Add labeled nodes
    node = LabeledNode(
        node_id="node_0",
        doc_id="doc_123",
        level=0,
        text="Chunk text...",
        score=15.0,
        confidence=0.95,  # Optional, defaults to 1.0
    )
    tree.add_node(node)

    # Save/load
    tree.save(Path("doc_123_labels.json"))
    loaded = LabeledTree.load(Path("doc_123_labels.json"))
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LabeledNode:
    """
    A node in the tree with a label (score).

    Labels can come from any source: ground truth data, oracle models,
    LLMs, or human annotation. The source is not distinguished--all are
    treated uniformly as labels.

    Supports hierarchical structure where internal nodes represent
    merged content from child nodes.

    Attributes:
        node_id: Unique identifier for this node
        doc_id: Parent document identifier
        level: Tree depth (0 = leaf chunk, 1+ = internal merge node)
        text: The node's content
        score: The label value (from any source)
        dimension_scores: Optional multi-dimensional scores
        reasoning: Rationale for the score (if available)
        confidence: Confidence in the label (0-1, defaults to 1.0)
        left_child_id: Left child for merge nodes
        right_child_id: Right child for merge nodes
        metadata: Task-specific fields (e.g., left_indicators for a task)
        timestamp: When this node was labeled
    """
    node_id: str
    doc_id: str
    level: int
    text: str
    score: float

    dimension_scores: Optional[Dict[str, float]] = None
    reasoning: str = ""
    confidence: float = 1.0

    left_child_id: Optional[str] = None
    right_child_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "doc_id": self.doc_id,
            "level": self.level,
            "text": self.text,
            "score": self.score,
            "dimension_scores": self.dimension_scores,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "left_child_id": self.left_child_id,
            "right_child_id": self.right_child_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabeledNode':
        """Create from dictionary."""
        return cls(
            node_id=data.get("node_id", ""),
            doc_id=data.get("doc_id", ""),
            level=data.get("level", 0),
            text=data.get("text", ""),
            score=data.get("score", 0.0),
            dimension_scores=data.get("dimension_scores"),
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 1.0),
            left_child_id=data.get("left_child_id"),
            right_child_id=data.get("right_child_id"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


@dataclass
class LabeledTree:
    """
    Complete labeled tree for a document.

    Stores labels (scores) for all nodes (leaves and internal merges)
    to enable testing all three OPS laws with known labels. Labels can
    come from any source (ground truth data, oracle models, LLMs, human
    annotation) and are treated uniformly.

    Attributes:
        doc_id: Document identifier
        document_text: Full document text
        document_score: Label for the full document
        nodes: Dict mapping node_id to LabeledNode
        levels: List of node_ids at each level (levels[0] = leaves)
        num_chunks: Total number of nodes
        num_levels: Number of tree levels
        metadata: Task-specific fields
        created_at: When this tree was created
        label_source: Model or source used to generate labels (optional)
    """
    doc_id: str
    document_text: str
    document_score: float

    nodes: Dict[str, LabeledNode] = field(default_factory=dict)
    levels: List[List[str]] = field(default_factory=list)

    num_chunks: int = 0
    num_levels: int = 0

    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    label_source: str = ""

    def add_node(self, node: LabeledNode):
        """Add a node to the tree."""
        self.nodes[node.node_id] = node

        # Ensure levels list is long enough
        while len(self.levels) <= node.level:
            self.levels.append([])

        # Add to appropriate level
        if node.node_id not in self.levels[node.level]:
            self.levels[node.level].append(node.node_id)

        # Update statistics
        self.num_chunks = len(self.nodes)
        self.num_levels = len(self.levels)

    def get_node(self, node_id: str) -> Optional[LabeledNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_level(self, level: int) -> List[LabeledNode]:
        """Get all nodes at a specific level."""
        if level >= len(self.levels):
            return []
        return [self.nodes[node_id] for node_id in self.levels[level]]

    def get_leaves(self) -> List[LabeledNode]:
        """Get all leaf nodes (level 0)."""
        return self.get_level(0)

    def get_merge_nodes(self) -> List[LabeledNode]:
        """Get all internal merge nodes (level > 0)."""
        merge_nodes = []
        for level in range(1, self.num_levels):
            merge_nodes.extend(self.get_level(level))
        return merge_nodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": "3.0",
            "doc_id": self.doc_id,
            "document_text": self.document_text,
            "document_score": self.document_score,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "levels": self.levels,
            "num_chunks": self.num_chunks,
            "num_levels": self.num_levels,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "label_source": self.label_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabeledTree':
        """Create from dictionary."""
        tree = cls(
            doc_id=data.get("doc_id", ""),
            document_text=data.get("document_text", ""),
            document_score=data.get("document_score", 0.0),
            levels=data.get("levels", []),
            num_chunks=data.get("num_chunks", 0),
            num_levels=data.get("num_levels", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            label_source=data.get("label_source", ""),
        )

        # Reconstruct nodes
        for node_id, node_data in data.get("nodes", {}).items():
            tree.nodes[node_id] = LabeledNode.from_dict(node_data)

        return tree

    def save(self, path: Path):
        """Save tree to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved labeled tree with {self.num_chunks} nodes to {path}")

    @classmethod
    def load(cls, path: Path) -> 'LabeledTree':
        """Load tree from JSON file."""
        with open(path) as f:
            data = json.load(f)

        tree = cls.from_dict(data)
        logger.info(f"Loaded labeled tree with {tree.num_chunks} nodes from {path}")
        return tree

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the tree."""
        if not self.nodes:
            return {"num_chunks": 0, "num_levels": 0}

        scores = [node.score for node in self.nodes.values()]

        return {
            "num_chunks": self.num_chunks,
            "num_levels": self.num_levels,
            "num_leaves": len(self.get_leaves()),
            "num_merge_nodes": len(self.get_merge_nodes()),
            "score_mean": sum(scores) / len(scores),
            "score_min": min(scores),
            "score_max": max(scores),
            "document_score": self.document_score,
            "label_source": self.label_source,
        }


class LabeledDataset:
    """
    Collection of labeled trees for multiple documents.

    Enables batch operations and statistics across documents.
    """

    def __init__(self, trees: Optional[List[LabeledTree]] = None):
        self.trees: Dict[str, LabeledTree] = {}
        if trees:
            for tree in trees:
                self.trees[tree.doc_id] = tree

    def add_tree(self, tree: LabeledTree):
        """Add a tree to the dataset."""
        self.trees[tree.doc_id] = tree

    def get_tree(self, doc_id: str) -> Optional[LabeledTree]:
        """Get a tree by document ID."""
        return self.trees.get(doc_id)

    def __len__(self) -> int:
        return len(self.trees)

    def __iter__(self):
        return iter(self.trees.values())

    def save(self, directory: Path):
        """Save all trees to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for doc_id, tree in self.trees.items():
            # Sanitize doc_id for filename
            safe_id = doc_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_id}_labels.json"
            tree.save(directory / filename)

        # Save index
        index = {
            "version": "3.0",
            "num_trees": len(self.trees),
            "doc_ids": list(self.trees.keys()),
            "created_at": datetime.now().isoformat(),
        }
        with open(directory / "index.json", 'w') as f:
            json.dump(index, f, indent=2)

        logger.info(f"Saved {len(self.trees)} labeled trees to {directory}")

    @classmethod
    def load(cls, directory: Path) -> 'LabeledDataset':
        """Load all trees from a directory."""
        directory = Path(directory)

        # Load index
        with open(directory / "index.json") as f:
            index = json.load(f)

        doc_ids = index.get("doc_ids", [])

        # Load all trees
        dataset = cls()
        for doc_id in doc_ids:
            safe_id = doc_id.replace("/", "_").replace("\\", "_")
            filename = f"{safe_id}_labels.json"
            filepath = directory / filename

            if filepath.exists():
                tree = LabeledTree.load(filepath)
                dataset.add_tree(tree)
            else:
                logger.warning(f"Could not find label file for {doc_id}: {filepath}")

        logger.info(f"Loaded {len(dataset)} labeled trees from {directory}")
        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics across all trees."""
        if not self.trees:
            return {"num_trees": 0}

        total_chunks = sum(tree.num_chunks for tree in self.trees.values())
        total_leaves = sum(len(tree.get_leaves()) for tree in self.trees.values())
        total_merges = sum(len(tree.get_merge_nodes()) for tree in self.trees.values())

        return {
            "num_trees": len(self.trees),
            "total_chunks": total_chunks,
            "total_leaves": total_leaves,
            "total_merge_nodes": total_merges,
            "avg_chunks_per_tree": total_chunks / len(self.trees),
            "avg_levels": sum(tree.num_levels for tree in self.trees.values()) / len(self.trees),
        }
