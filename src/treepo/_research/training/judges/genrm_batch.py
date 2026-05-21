"""Judge-facing wrapper over the internal batched GenRM client."""

from treepo._research.training.preference.genrm_batch import (
    AsyncBatchGenRMClient,
    GenRMComparisonRequest,
    create_genrm_batch_client,
)

__all__ = [
    "AsyncBatchGenRMClient",
    "GenRMComparisonRequest",
    "create_genrm_batch_client",
]
