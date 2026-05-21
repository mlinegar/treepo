"""
Data sources for unified preference collection.

Provides a protocol-based abstraction for different data input types,
enabling the unified preference collection script to work with:
- Direct documents from task data loaders
- Pre-generated labeled trees (with oracle scores at each node)
- Synthetic data files

Example usage:
    # Direct documents
    source = DirectDocumentSource(
        task=get_task("document_analysis"),
        max_documents=100,
    )
    for example in source.get_examples():
        # process example
        pass

    # Labeled trees
    source = LabeledTreeSource(
        labels_dir=Path("data/labels"),
        law_type="sufficiency",
    )
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from treepo._research.tasks.base import AbstractTask

logger = logging.getLogger(__name__)


@dataclass
class DataSourceExample:
    """
    A single example from a data source for preference collection.

    This is the common data structure that all data sources yield,
    enabling uniform processing across different input types.
    """

    example_id: str
    """Unique identifier for this example."""

    text: str
    """The text content to summarize."""

    rubric: str
    """The preservation rubric for summarization."""

    reference_score: Optional[float] = None
    """Optional reference score (oracle label) for this text."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata (e.g., source file, tree info)."""

    # For merge law - child summaries to merge
    child_summaries: Optional[Tuple[str, str]] = None
    """For merge law: left and right child summaries to merge."""

    # For idempotence law - original summary to re-summarize
    original_summary: Optional[str] = None
    """For idempotence law: the summary to re-summarize."""


@runtime_checkable
class PreferenceDataSource(Protocol):
    """
    Protocol for data sources that provide examples for preference collection.

    Implementations must provide:
    - source_name: Unique identifier for the source type
    - get_examples(): Iterator of DataSourceExample objects
    - get_rubric(): The rubric to use for this data source
    """

    @property
    def source_name(self) -> str:
        """Unique identifier for this data source type."""
        ...

    def get_examples(self) -> Iterator[DataSourceExample]:
        """Yield examples for preference collection."""
        ...

    def get_rubric(self) -> str:
        """Get the preservation rubric for this data source."""
        ...


class DirectDocumentSource:
    """
    Load documents directly from a task's data loader.

    Uses the task's data loader to get documents with their ground truth scores.

    Example:
        task = get_task("document_analysis")
        source = DirectDocumentSource(task, max_documents=100)
        for example in source.get_examples():
            print(example.example_id, example.reference_score)
    """

    def __init__(
        self,
        task: "AbstractTask",
        max_documents: Optional[int] = None,
        splits: Optional[List[str]] = None,
    ):
        """
        Initialize direct document source.

        Args:
            task: AbstractTask instance
            max_documents: Maximum documents to return (None = all)
            splits: Which splits to include (default: ["train", "val"])
        """
        self.task = task
        self.max_documents = max_documents
        self.splits = splits or ["train", "val"]
        self._rubric: Optional[str] = None

    @property
    def source_name(self) -> str:
        return f"direct_{self.task.name}"

    def get_rubric(self) -> str:
        """Get rubric from the task."""
        if self._rubric is None:
            self._rubric = self.task.create_rubric()
        return self._rubric

    def get_examples(self) -> Iterator[DataSourceExample]:
        """Yield examples from the task's data loader."""
        samples = self._load_samples()

        if self.max_documents:
            samples = samples[: self.max_documents]

        rubric = self.get_rubric()

        def get_value(item: Any, key: str, default: Any = None) -> Any:
            if isinstance(item, dict):
                return item.get(key, default)
            return getattr(item, key, default)

        for i, sample in enumerate(samples):
            doc_id = get_value(sample, self.task.id_field, None)
            if doc_id is None:
                doc_id = get_value(sample, "id", f"doc_{i}")

            doc_text = (
                get_value(sample, self.task.text_field, "") or
                get_value(sample, "text", "") or
                get_value(sample, "content", "")
            )

            if not doc_text:
                logger.warning(f"Skipping document {doc_id}: no text")
                continue
            ground_truth = get_value(sample, self.task.label_field, None)
            if ground_truth is None:
                ground_truth = get_value(sample, "reference_score", None)
            if ground_truth is None:
                ground_truth = get_value(sample, "score", None)
            if ground_truth is not None and hasattr(self.task, "normalize_score"):
                try:
                    ground_truth = self.task.normalize_score(ground_truth)
                except Exception:
                    pass

            yield DataSourceExample(
                example_id=doc_id,
                text=doc_text,
                rubric=rubric,
                reference_score=ground_truth,
                metadata={
                    "source": "direct",
                    "split": get_value(sample, "split", "unknown"),
                    "original_length": len(doc_text),
                },
            )

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load samples from the task's data loader."""
        # Use task's get_samples() method - all tasks should implement this
        if hasattr(self.task, "get_samples"):
            return self.task.get_samples(splits=self.splits)

        raise NotImplementedError(
            f"No data loader implemented for task '{self.task.name}'. "
            f"Implement task.get_samples() method."
        )


class LabeledTreeSource:
    """
    Load examples from pre-generated labeled trees.

    Labeled trees have oracle scores (labels) at all levels, enabling
    preference collection for all three OPS laws:
    - Sufficiency: Leaf nodes (original chunks)
    - Idempotence: Re-summarize leaf summaries
    - Merge: Parent nodes (merge children)

    Example:
        source = LabeledTreeSource(
            labels_dir=Path("data/labels"),
            law_type="merge",
            max_trees=10,
        )
        for example in source.get_examples():
            if example.child_summaries:
                # This is a merge example
                left, right = example.child_summaries
    """

    def __init__(
        self,
        labels_dir: Path,
        law_type: str = "sufficiency",
        max_trees: Optional[int] = None,
        max_nodes_per_tree: Optional[int] = None,
        rubric: Optional[str] = None,
    ):
        """
        Initialize labeled tree source.

        Args:
            labels_dir: Directory containing labeled tree files
            law_type: Which OPS law to collect for (sufficiency/idempotence/merge)
            max_trees: Maximum trees to process
            max_nodes_per_tree: Maximum nodes per tree
            rubric: Optional rubric override
        """
        self.labels_dir = Path(labels_dir)
        self.law_type = law_type
        self.max_trees = max_trees
        self.max_nodes_per_tree = max_nodes_per_tree
        self._rubric = rubric
        self._dataset = None

    @property
    def source_name(self) -> str:
        return f"labeled_{self.law_type}"

    def get_rubric(self) -> str:
        """Get rubric for labeled tree source."""
        if self._rubric is None:
            # Use generic rubric - task-specific rubrics should be passed in
            self._rubric = (
                "Preserve key information from the original text. "
                "The summary should enable accurate downstream analysis."
            )
        return self._rubric

    def _load_dataset(self):
        """Load labeled dataset lazily."""
        if self._dataset is None:
            from treepo._research.training.tree import (
                LabeledDataset,
            )

            self._dataset = LabeledDataset.load(self.labels_dir)
            logger.info(f"Loaded {len(self._dataset)} labeled trees")
        return self._dataset

    def get_examples(self) -> Iterator[DataSourceExample]:
        """Yield examples from labeled trees."""
        dataset = self._load_dataset()
        trees = list(dataset.trees.values())

        if self.max_trees:
            trees = trees[: self.max_trees]

        rubric = self.get_rubric()

        for tree in trees:
            # Get nodes based on law type
            if self.law_type == "merge":
                nodes = tree.get_merge_nodes()
            else:
                # sufficiency and idempotence use leaf nodes
                nodes = tree.get_leaves()

            if self.max_nodes_per_tree:
                nodes = nodes[: self.max_nodes_per_tree]

            for node in nodes:
                metadata = {
                    "source": "labeled",
                    "tree_id": tree.doc_id,
                    "node_level": node.level,
                    "law_type": self.law_type,
                }

                # Build example based on law type
                if self.law_type == "merge":
                    # For merge, include child summaries
                    left_child = getattr(node, "left_child", None)
                    right_child = getattr(node, "right_child", None)

                    child_summaries = None
                    if left_child and right_child:
                        left_text = getattr(left_child, "summary", None) or getattr(
                            left_child, "text", ""
                        )
                        right_text = getattr(right_child, "summary", None) or getattr(
                            right_child, "text", ""
                        )
                        child_summaries = (left_text, right_text)

                    yield DataSourceExample(
                        example_id=node.node_id,
                        text=node.text,
                        rubric=rubric,
                        reference_score=node.score,
                        metadata=metadata,
                        child_summaries=child_summaries,
                    )

                else:
                    # sufficiency or idempotence - just the node text
                    yield DataSourceExample(
                        example_id=node.node_id,
                        text=node.text,
                        rubric=rubric,
                        reference_score=node.score,
                        metadata=metadata,
                    )


class SyntheticDataSource:
    """
    Load examples from synthetic data files.

    Supports JSONL and JSON formats with flexible field mapping.

    Expected format (each line for JSONL, or list items for JSON):
    {
        "id": "example_001",
        "text": "The document text...",
        "score": 25.0,  // optional ground truth
        ...
    }

    Example:
        source = SyntheticDataSource(
            data_path=Path("data/synthetic.jsonl"),
            rubric="Preserve key information...",
        )
    """

    def __init__(
        self,
        data_path: Path,
        rubric: Optional[str] = None,
        id_field: str = "id",
        text_field: str = "text",
        score_field: str = "score",
        max_examples: Optional[int] = None,
    ):
        """
        Initialize synthetic data source.

        Args:
            data_path: Path to JSONL or JSON file
            rubric: Preservation rubric (required)
            id_field: Field name for example ID
            text_field: Field name for text content
            score_field: Field name for ground truth score
            max_examples: Maximum examples to return
        """
        self.data_path = Path(data_path)
        self._rubric = rubric
        self.id_field = id_field
        self.text_field = text_field
        self.score_field = score_field
        self.max_examples = max_examples

    @property
    def source_name(self) -> str:
        return f"synthetic_{self.data_path.stem}"

    def get_rubric(self) -> str:
        """Get rubric for synthetic source."""
        if self._rubric is None:
            raise ValueError("Rubric must be provided for SyntheticDataSource")
        return self._rubric

    def get_examples(self) -> Iterator[DataSourceExample]:
        """Yield examples from synthetic data file."""
        rubric = self.get_rubric()
        count = 0

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path, "r") as f:
                for line_num, line in enumerate(f):
                    if self.max_examples and count >= self.max_examples:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        example = self._record_to_example(record, line_num, rubric)
                        if example:
                            yield example
                            count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num + 1}: {e}")

        elif self.data_path.suffix == ".json":
            with open(self.data_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                for idx, record in enumerate(data):
                    if self.max_examples and count >= self.max_examples:
                        break

                    example = self._record_to_example(record, idx, rubric)
                    if example:
                        yield example
                        count += 1
            else:
                logger.error("JSON file must contain a list of records")

        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

    def _record_to_example(
        self,
        record: Dict[str, Any],
        index: int,
        rubric: str,
    ) -> Optional[DataSourceExample]:
        """Convert a record to a DataSourceExample."""
        example_id = record.get(self.id_field, f"synthetic_{index}")
        text = record.get(self.text_field, "")

        if not text:
            logger.warning(f"Skipping record {example_id}: no text")
            return None


        ground_truth = record.get(self.score_field)

        # Include all other fields in metadata
        metadata = {
            "source": "synthetic",
            "file": str(self.data_path),
            "index": index,
        }
        for key, value in record.items():
            if key not in (self.id_field, self.text_field, self.score_field):
                metadata[key] = value

        return DataSourceExample(
            example_id=example_id,
            text=text,
            rubric=rubric,
            reference_score=ground_truth,
            metadata=metadata,
        )


def create_data_source(
    source_type: str,
    task: Optional["AbstractTask"] = None,
    **kwargs,
) -> PreferenceDataSource:
    """
    Factory function to create a data source.

    Args:
        source_type: One of "direct", "labeled", "synthetic"
        task: AbstractTask instance (required for "direct" source)
        **kwargs: Additional arguments passed to the source constructor

    Returns:
        PreferenceDataSource instance

    Example:
        source = create_data_source(
            "direct",
            task=get_task("document_analysis"),
            max_documents=100,
        )
    """
    if source_type == "direct":
        if task is None:
            raise ValueError("task is required for direct source")
        return DirectDocumentSource(task=task, **kwargs)

    elif source_type == "labeled":
        labels_dir = kwargs.pop("labels_dir", None)
        if labels_dir is None:
            raise ValueError("labels_dir is required for labeled source")
        return LabeledTreeSource(labels_dir=labels_dir, **kwargs)

    elif source_type == "synthetic":
        data_path = kwargs.pop("data_path", None) or kwargs.pop("synthetic_data_path", None)
        if data_path is None:
            raise ValueError("data_path is required for synthetic source")
        return SyntheticDataSource(data_path=data_path, **kwargs)

    else:
        raise ValueError(f"Unknown source type: {source_type}")
