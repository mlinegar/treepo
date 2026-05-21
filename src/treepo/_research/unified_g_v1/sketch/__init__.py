"""Minimal bigram-sketch synthetic setting.

Mirrors the Markov synthetic setting as a second "basic" canonical case for
the unified fit() pipeline. The sketch representation follows the Lean
formalization in `lean3/FormalProofs/OPT/BigramSketch.lean`:

  sketch(xs) = (bigram_counts, first_token, last_token, length)
  merge(sx, sy) = bigram_counts_x + bigram_counts_y + boundary(last_x, first_y),
                  first = first_x, last = last_y, length = length_x + length_y

The synthetic training task: a learned MLP merge operator should recover this
analytic merge; a linear head regresses a scalar target derived from root
bigram counts.
"""
from treepo._research.unified_g_v1.sketch.classical_parity import (
    ClassicalHLLParityConfig,
    ClassicalHLLParityOracle,
    classical_hll_parity_task,
    run_classical_sketch_baseline,
)
from treepo._research.unified_g_v1.sketch.classical_sketch_grid import (
    classical_sketch_grid_task,
    run_classical_sketch_grid_baseline,
)
from treepo._research.unified_g_v1.sketch.hll_precision_floor import (
    HLLPrecisionFloorConfig,
    plot_precision_floor_recovery,
    run_precision_floor_cell,
    run_precision_floor_sweep,
    write_precision_floor_outputs,
)
from treepo._research.unified_g_v1.sketch.learned_additive_state import (
    ExactNumericStateSpec,
    LearnedAdditiveStateConfig,
    LearnedAdditiveStateObjective,
    LearnedAdditiveStateOracle,
    OracleStateAdditiveMergeModel,
    exact_numeric_leaf_state,
    exact_numeric_merge_state,
    exact_numeric_readout,
    exact_numeric_state_spec,
    learned_additive_state_task,
)
from treepo._research.unified_g_v1.sketch.learned_hll_parity import (
    LearnedHLLMergeModel,
    LearnedHLLParityObjective,
    LearnedHLLParityOracle,
    OracleStateHLLMergeModel,
    hll_estimate_differentiable,
    learned_hll_parity_task,
)
from treepo._research.unified_g_v1.sketch.learned_scalar_sketch import (
    LearnedScalarSketchConfig,
    LearnedScalarSketchMergeModel,
    LearnedScalarSketchObjective,
    LearnedScalarSketchOracle,
    learned_method_name,
    learned_scalar_sketch_task,
    learned_sketch_sequence_task,
    learned_variant_codename,
    load_final_sketch_models,
    sequenced_learned_sketch_trainer,
)
from treepo._research.unified_g_v1.sketch.learned_sketch_grid import (
    LearnedGridTarget,
    learned_grid_targets,
    run_learned_sketch_grid,
)
from treepo._research.unified_g_v1.sketch.runner import (
    SketchRunRecord,
    SketchRunSpec,
    run_sketch_spec,
)

__all__ = [
    "ClassicalHLLParityConfig",
    "ClassicalHLLParityOracle",
    "LearnedHLLMergeModel",
    "LearnedHLLParityObjective",
    "LearnedHLLParityOracle",
    "OracleStateHLLMergeModel",
    "LearnedScalarSketchConfig",
    "LearnedScalarSketchMergeModel",
    "LearnedScalarSketchObjective",
    "LearnedScalarSketchOracle",
    "LearnedAdditiveStateConfig",
    "LearnedAdditiveStateObjective",
    "LearnedAdditiveStateOracle",
    "OracleStateAdditiveMergeModel",
    "ExactNumericStateSpec",
    "LearnedGridTarget",
    "HLLPrecisionFloorConfig",
    "SketchRunRecord",
    "SketchRunSpec",
    "classical_hll_parity_task",
    "classical_sketch_grid_task",
    "hll_estimate_differentiable",
    "learned_hll_parity_task",
    "learned_additive_state_task",
    "exact_numeric_leaf_state",
    "exact_numeric_merge_state",
    "exact_numeric_readout",
    "exact_numeric_state_spec",
    "learned_grid_targets",
    "learned_method_name",
    "learned_scalar_sketch_task",
    "learned_sketch_sequence_task",
    "learned_variant_codename",
    "load_final_sketch_models",
    "plot_precision_floor_recovery",
    "run_precision_floor_cell",
    "run_precision_floor_sweep",
    "sequenced_learned_sketch_trainer",
    "write_precision_floor_outputs",
    "run_classical_sketch_baseline",
    "run_classical_sketch_grid_baseline",
    "run_learned_sketch_grid",
    "run_sketch_spec",
]
