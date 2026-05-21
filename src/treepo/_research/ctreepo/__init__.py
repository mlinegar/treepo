"""C-TreePO package layer for ThinkingTrees.

This namespace provides a stable surface for C-TreePO simulations and tooling.
It is designed to be extractable into a standalone package later.
"""

from __future__ import annotations

from treepo._research.ctreepo.contracts import (  # noqa: E402
    ArtifactRef,
    CTreePOFitResult,
    CTreePOLearningSpec,
    CTreePOProgramSpec,
    RunManifest,
)

__all__ = [
    "__version__",
    "ArtifactRef",
    "CTreePOFitResult",
    "CTreePOLearningSpec",
    "CTreePOProgramSpec",
    "RunManifest",
]

# Keep a separate version so ctreepo can be split out later without breaking callers.
__version__ = "0.1.0"
