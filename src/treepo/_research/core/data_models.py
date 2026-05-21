"""
Core data models for OPS (Oracle-Preserving Summarization) trees.

This module defines the fundamental data structures:
- Node: Individual nodes in the summarization tree
- Tree: Container for the complete tree structure
"""

from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Callable, Any, Dict
from enum import Enum
from pathlib import Path
import uuid
import json


class AuditStatus(Enum):
    """Status of node audit verification."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class AuditResult:
    """Result of an audit check on a node."""
    status: AuditStatus
    discrepancy_score: float = 0.0
    reasoning: Optional[str] = None
    trace: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit result to dictionary."""
        return {
            'status': self.status.value,
            'discrepancy_score': self.discrepancy_score,
            'reasoning': self.reasoning,
            'trace': self.trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditResult':
        """Deserialize audit result from dictionary."""
        return cls(
            status=AuditStatus(data['status']),
            discrepancy_score=data.get('discrepancy_score', 0.0),
            reasoning=data.get('reasoning'),
            trace=data.get('trace'),
        )


@dataclass
class Node:
    """
    A node in the OPS (Oracle-Preserving Summarization) tree.

    Leaves contain raw text spans from the original document.
    Internal nodes contain summaries of their children.

    Attributes:
        id: Unique identifier for this node
        level: Depth in tree (0 = leaf)
        raw_text_span: Original text (only for leaves)
        summary: The summary text at this node
        left_child: Left child node (None for leaves)
        right_child: Right child node (None for leaves)
        parent: Parent node (None for root)
        audit_result: Result of verification audit
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: int = 0

    # Content
    raw_text_span: Optional[str] = None
    ops_span: Optional[str] = None
    summary: str = ""

    # Structure
    left_child: Optional['Node'] = None
    right_child: Optional['Node'] = None
    parent: Optional['Node'] = None

    # Audit state
    audit_result: AuditResult = field(
        default_factory=lambda: AuditResult(status=AuditStatus.PENDING)
    )
    # Per-node auxiliary payload (support spans, embeddings, sketch states, etc.).
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return self.left_child is None and self.right_child is None

    @property
    def is_root(self) -> bool:
        """Check if this node is the root (no parent)."""
        return self.parent is None

    @property
    def has_both_children(self) -> bool:
        """Check if node has both left and right children."""
        return self.left_child is not None and self.right_child is not None

    @property
    def children(self) -> List['Node']:
        """Get list of children (0, 1, or 2 nodes)."""
        result = []
        if self.left_child is not None:
            result.append(self.left_child)
        if self.right_child is not None:
            result.append(self.right_child)
        return result

    @property
    def audit_passed(self) -> bool:
        """Check if audit passed."""
        return self.audit_result.status == AuditStatus.PASSED

    @property
    def discrepancy_score(self) -> float:
        """Get the discrepancy score from audit."""
        return self.audit_result.discrepancy_score

    def set_audit_passed(self, score: float = 0.0, reasoning: str = "") -> None:
        """Mark this node as having passed audit."""
        self.audit_result = AuditResult(
            status=AuditStatus.PASSED,
            discrepancy_score=score,
            reasoning=reasoning
        )

    def set_audit_failed(self, score: float, reasoning: str = "") -> None:
        """Mark this node as having failed audit."""
        self.audit_result = AuditResult(
            status=AuditStatus.FAILED,
            discrepancy_score=score,
            reasoning=reasoning
        )

    def validate(self) -> List[str]:
        """
        Check node invariants and return list of violations.

        Returns:
            List of violation descriptions (empty if valid)
        """
        violations = []

        # Leaf invariants
        if self.is_leaf:
            if self.level != 0:
                violations.append(f"Leaf node has non-zero level: {self.level}")
        else:
            # Internal node invariants
            if self.level == 0:
                violations.append("Internal node has level 0")
            if self.raw_text_span is not None:
                violations.append("Internal node has raw_text_span set")

        # Binary tree constraint: both children or neither
        has_left = self.left_child is not None
        has_right = self.right_child is not None
        if has_left != has_right:
            violations.append("Node has exactly one child (must have 0 or 2)")

        # Parent-child consistency
        for child in self.children:
            if child.parent is not self:
                violations.append(f"Child {child.id} doesn't reference this node as parent")

        return violations

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize node to dictionary.

        Uses child IDs instead of full child objects to avoid circular references.
        Parent reference is excluded (will be reconstructed during from_dict).
        """
        return {
            'id': self.id,
            'level': self.level,
            'raw_text_span': self.raw_text_span,
            'ops_span': self.ops_span,
            'summary': self.summary,
            'left_child_id': self.left_child.id if self.left_child else None,
            'right_child_id': self.right_child.id if self.right_child else None,
            'audit_result': self.audit_result.to_dict(),
            'metadata': dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """
        Deserialize node from dictionary.

        Note: Child and parent references must be rebuilt separately
        after all nodes are created. Use Tree.from_dict() for full
        tree reconstruction.
        """
        node = cls(
            id=data['id'],
            level=data['level'],
            raw_text_span=data.get('raw_text_span'),
            ops_span=data.get('ops_span'),
            summary=data.get('summary', ''),
            audit_result=AuditResult.from_dict(data['audit_result']),
            metadata=dict(data.get('metadata', {}) or {}),
        )
        # Store child IDs for later linking (not actual references yet)
        node._left_child_id = data.get('left_child_id')
        node._right_child_id = data.get('right_child_id')
        return node

    def __repr__(self) -> str:
        node_type = "Leaf" if self.is_leaf else "Internal"
        summary_preview = self.summary[:30] + "..." if len(self.summary) > 30 else self.summary
        return f"Node({node_type}, id={self.id}, level={self.level}, summary='{summary_preview}')"


@dataclass
class Tree:
    """
    Container for an OPS (Oracle-Preserving Summarization) tree.

    The tree is built bottom-up from document chunks (leaves) through
    recursive summarization to a single root node.

    Attributes:
        root: The root node containing the final summary
        rubric: Information preservation criteria for summarization
        metadata: Additional information about source document
    """
    root: Node
    rubric: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Get the height of the tree (max depth from root to leaf)."""
        if self.root is None:
            return 0
        return self._calculate_height(self.root)

    def _calculate_height(self, node: Node) -> int:
        """Recursively calculate height from a node."""
        if node.is_leaf:
            return 0
        left_height = self._calculate_height(node.left_child) if node.left_child else 0
        right_height = self._calculate_height(node.right_child) if node.right_child else 0
        return 1 + max(left_height, right_height)

    @property
    def node_count(self) -> int:
        """Get total number of nodes in the tree."""
        return len(list(self.traverse_preorder()))

    @property
    def leaf_count(self) -> int:
        """Get number of leaf nodes."""
        return len(self.leaves)

    @property
    def leaves(self) -> List[Node]:
        """Get all leaf nodes in left-to-right order."""
        return [node for node in self.traverse_inorder() if node.is_leaf]

    @property
    def internal_nodes(self) -> List[Node]:
        """Get all internal (non-leaf) nodes."""
        return [node for node in self.traverse_preorder() if not node.is_leaf]

    @property
    def final_summary(self) -> str:
        """Get the root summary (final output)."""
        return self.root.summary if self.root else ""

    @property
    def audit_failure_rate(self) -> float:
        """Calculate proportion of failed audits."""
        all_nodes = list(self.traverse_preorder())
        if not all_nodes:
            return 0.0
        failed = sum(1 for n in all_nodes if n.audit_result.status == AuditStatus.FAILED)
        return failed / len(all_nodes)

    def traverse_preorder(self) -> Iterator[Node]:
        """Traverse tree in preorder (root, left, right)."""
        if self.root is None:
            return
        yield from self._preorder(self.root)

    def _preorder(self, node: Node) -> Iterator[Node]:
        """Helper for preorder traversal."""
        yield node
        if node.left_child:
            yield from self._preorder(node.left_child)
        if node.right_child:
            yield from self._preorder(node.right_child)

    def traverse_postorder(self) -> Iterator[Node]:
        """Traverse tree in postorder (left, right, root)."""
        if self.root is None:
            return
        yield from self._postorder(self.root)

    def _postorder(self, node: Node) -> Iterator[Node]:
        """Helper for postorder traversal."""
        if node.left_child:
            yield from self._postorder(node.left_child)
        if node.right_child:
            yield from self._postorder(node.right_child)
        yield node

    def traverse_inorder(self) -> Iterator[Node]:
        """Traverse tree in inorder (left, root, right)."""
        if self.root is None:
            return
        yield from self._inorder(self.root)

    def _inorder(self, node: Node) -> Iterator[Node]:
        """Helper for inorder traversal."""
        if node.left_child:
            yield from self._inorder(node.left_child)
        yield node
        if node.right_child:
            yield from self._inorder(node.right_child)

    def traverse_level_order(self) -> Iterator[Node]:
        """Traverse tree in level order (BFS)."""
        if self.root is None:
            return
        from collections import deque
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            yield node
            if node.left_child:
                queue.append(node.left_child)
            if node.right_child:
                queue.append(node.right_child)

    def find_node(self, node_id: str) -> Optional[Node]:
        """Find a node by its ID."""
        for node in self.traverse_preorder():
            if node.id == node_id:
                return node
        return None

    def get_path_to_root(self, node: Node) -> List[Node]:
        """Get path from a node to the root."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def get_failed_audits(self) -> List[Node]:
        """Get all nodes that failed audit."""
        return [
            node for node in self.traverse_preorder()
            if node.audit_result.status == AuditStatus.FAILED
        ]

    def validate(self) -> List[str]:
        """
        Validate entire tree structure.

        Returns:
            List of violation descriptions (empty if valid)
        """
        violations = []

        if self.root is None:
            violations.append("Tree has no root")
            return violations

        # Check each node
        for node in self.traverse_preorder():
            node_violations = node.validate()
            for v in node_violations:
                violations.append(f"Node {node.id}: {v}")

        # Check that root has no parent
        if self.root.parent is not None:
            violations.append("Root node has a parent")

        # Check level consistency (children must have lower levels than parent)
        # Note: We allow children at any lower level (not just level-1) to
        # accommodate odd nodes that get promoted during tree construction
        for node in self.traverse_preorder():
            if not node.is_leaf:
                for child in node.children:
                    if child.level >= node.level:
                        violations.append(
                            f"Level inconsistency: parent {node.id} (level {node.level}) "
                            f"has child {child.id} at same or higher level ({child.level})"
                        )

        return violations

    def apply_to_all(self, func: Callable[[Node], Any]) -> List[Any]:
        """Apply a function to all nodes and return results."""
        return [func(node) for node in self.traverse_preorder()]

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize tree to dictionary.

        Returns a flat list of nodes plus tree metadata. Node references
        are stored as IDs rather than nested objects.
        """
        nodes = [node.to_dict() for node in self.traverse_preorder()]
        return {
            'version': 1,
            'root_id': self.root.id if self.root else None,
            'rubric': self.rubric,
            'metadata': self.metadata,
            'nodes': nodes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Tree':
        """
        Deserialize tree from dictionary.

        Reconstructs all node references (children and parents).
        """
        # Create all nodes first (without links)
        nodes_by_id: Dict[str, Node] = {}
        for node_data in data['nodes']:
            node = Node.from_dict(node_data)
            nodes_by_id[node.id] = node

        # Rebuild child and parent references
        for node in nodes_by_id.values():
            left_id = getattr(node, '_left_child_id', None)
            right_id = getattr(node, '_right_child_id', None)

            if left_id and left_id in nodes_by_id:
                node.left_child = nodes_by_id[left_id]
                node.left_child.parent = node

            if right_id and right_id in nodes_by_id:
                node.right_child = nodes_by_id[right_id]
                node.right_child.parent = node

            # Clean up temporary attributes
            if hasattr(node, '_left_child_id'):
                delattr(node, '_left_child_id')
            if hasattr(node, '_right_child_id'):
                delattr(node, '_right_child_id')

        # Get root
        root_id = data.get('root_id')
        root = nodes_by_id.get(root_id) if root_id else None

        if root is None and nodes_by_id:
            # Fallback: find node with no parent
            for node in nodes_by_id.values():
                if node.parent is None:
                    root = node
                    break

        return cls(
            root=root,
            rubric=data.get('rubric', ''),
            metadata=data.get('metadata', {}),
        )

    def save(self, path: Path) -> None:
        """
        Save tree to JSON file.

        Args:
            path: File path (will be created/overwritten)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> 'Tree':
        """
        Load tree from JSON file.

        Args:
            path: File path to load from

        Returns:
            Reconstructed Tree
        """
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"Tree(height={self.height}, nodes={self.node_count}, "
            f"leaves={self.leaf_count}, rubric='{self.rubric[:30]}...')"
        )


def leaf(
    text: str,
    node_id: Optional[str] = None,
    summary: Optional[str] = None,
    require_summarization: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Node:
    """
    Factory function to create a leaf node.

    Per the OPS paper, leaf nodes should have their raw_text_span summarized
    through the summarizer g() before being used in the tree. This function
    allows two modes:

    1. summary=None, require_summarization=False: Uses raw text as placeholder
       summary (legacy behavior, but logs warning)
    2. summary=<text>: Uses provided summary (correct behavior per paper)
    3. summary=None, require_summarization=True: Raises error (strict mode)

    Args:
        text: The raw text for this leaf
        node_id: Optional custom ID
        summary: The summarized version of the text (should come from g())
        require_summarization: If True, raises error when summary is None

    Returns:
        A properly configured leaf node

    Raises:
        ValueError: If require_summarization=True and summary is None
    """
    import logging
    logger = logging.getLogger(__name__)

    if summary is None:
        if require_summarization:
            raise ValueError(
                "Leaf nodes must have a summary from the summarizer g(). "
                "Pass summary=<summarized_text> or set require_summarization=False."
            )
        # Legacy behavior: use raw text as summary (with warning)
        # This is inconsistent with the paper but allows backward compatibility
        logger.debug(
            f"Creating leaf node with raw text as summary. "
            f"Per OPS paper, leaves should be summarized through g()."
        )
        summary = text

    node = Node(
        level=0,
        raw_text_span=text,
        ops_span=text,
        summary=summary,
    )
    if node_id:
        node.id = node_id
    if metadata:
        node.metadata.update(dict(metadata))
    return node


def node(
    left: Node,
    right: Node,
    summary: str,
    node_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Node:
    """
    Factory function to create an internal node.

    Args:
        left: Left child node
        right: Right child node
        summary: Summary text for this node
        node_id: Optional custom ID

    Returns:
        A properly configured internal node with parent refs set
    """
    level = max(left.level, right.level) + 1
    from treepo._research.core.protocols import format_merge_input

    left_ops_span = left.ops_span or left.raw_text_span or left.summary or ""
    right_ops_span = right.ops_span or right.raw_text_span or right.summary or ""

    node = Node(
        level=level,
        ops_span=format_merge_input(left_ops_span, right_ops_span),
        summary=summary,
        left_child=left,
        right_child=right
    )
    if node_id:
        node.id = node_id

    # Set parent references
    left.parent = node
    right.parent = node

    if metadata:
        node.metadata.update(dict(metadata))

    # Propagate best-effort support span metadata (char offsets) when present.
    try:
        l_start = left.metadata.get("char_start")
        l_end = left.metadata.get("char_end")
        r_start = right.metadata.get("char_start")
        r_end = right.metadata.get("char_end")
        if all(isinstance(v, int) for v in [l_start, l_end, r_start, r_end]):
            node.metadata.setdefault("char_start", min(int(l_start), int(r_start)))
            node.metadata.setdefault("char_end", max(int(l_end), int(r_end)))
    except Exception:
        pass

    return node
