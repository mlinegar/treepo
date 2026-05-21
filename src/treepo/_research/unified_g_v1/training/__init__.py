from treepo._research.unified_g_v1.training.bundle_helper import run_training_bundle
from treepo._research.unified_g_v1.training.config_io import load_trainer_config
from treepo._research.unified_g_v1.training.component_ladder import (
    ComponentLadderResult,
    ComponentLadderStageContext,
    ComponentLadderStageOutput,
    ComponentLadderStageRecord,
    run_component_ladder,
)
from treepo._research.unified_g_v1.training.fit import (
    FitResult,
    PyTorchRuntime,
    TrainerConfig,
    TreeTaskConfig,
    fit,
)
from treepo._research.unified_g_v1.dimension_guards import (
    DimensionInvariantWarning,
    promote_dim,
    require_dim,
)
from treepo._research.unified_g_v1.training.prepared_dataset import (
    DATASET_MANIFEST_VERSION,
    MANIFEST_FILENAME,
    PreparedDataset,
    write_dataset_manifest,
)
from treepo._research.unified_g_v1.training.recipes import (
    manifesto_rile_embedding_fno_task,
    manifesto_rile_tree_dspy_task,
    manifesto_rile_text_llm_task,
    markov_task,
    mergeable_sketch_task,
)
from treepo._research.unified_g_v1.training.tree_task import (
    TreeExample,
    TreeObjective,
    TreeOracle,
    run_tree_task,
)
from treepo._research.unified_g_v1.training.trainers import (
    TRAINER_REGISTRY,
    Trainer,
    register_trainer,
    resolve_trainer,
)

__all__ = [
    "DATASET_MANIFEST_VERSION",
    "FitResult",
    "MANIFEST_FILENAME",
    "PreparedDataset",
    "PyTorchRuntime",
    "TRAINER_REGISTRY",
    "ComponentLadderResult",
    "ComponentLadderStageContext",
    "ComponentLadderStageOutput",
    "ComponentLadderStageRecord",
    "DimensionInvariantWarning",
    "Trainer",
    "TrainerConfig",
    "TreeExample",
    "TreeObjective",
    "TreeOracle",
    "TreeTaskConfig",
    "fit",
    "load_trainer_config",
    "manifesto_rile_embedding_fno_task",
    "manifesto_rile_tree_dspy_task",
    "manifesto_rile_text_llm_task",
    "markov_task",
    "mergeable_sketch_task",
    "promote_dim",
    "register_trainer",
    "require_dim",
    "run_component_ladder",
    "resolve_trainer",
    "run_training_bundle",
    "run_tree_task",
    "write_dataset_manifest",
]
