"""
Unified Training Loop for Judge and Generator Co-Optimization.

This module implements the closed-loop training system where both the judge
(GenRM) and the generator (summarizer) are iteratively improved together.

The loop:
1. Generate k candidates with current generator
2. Rank with current judge (GenRM by default) → preferences collected FREE
3. Select winner via tournament
4. Train generator using pluggable method (DPO, SFT, GRPO, BootstrapFinetune)
5. Update judge via GEPA on oracle-enriched preferences
6. Repeat with updated generator AND updated judge

This implements a co-optimization loop where both components improve together,
with the training signal coming from downstream oracle performance.

Usage:
    from treepo._research.training.unified_trainer import UnifiedTrainer, UnifiedTrainerConfig
    from treepo._research.training.generator_trainers import DPOGeneratorTrainer

    trainer = UnifiedTrainer(
        generator_trainer=DPOGeneratorTrainer(),
        genrm_judge=judge,
        oracle_predict=oracle.predict,
        config=UnifiedTrainerConfig(max_iterations=5),
        output_dir="outputs/unified",
    )

    result = trainer.train(samples, rubric)
    print(f"Final judge accuracy: {result.final_judge_accuracy:.3f}")
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union

from treepo._research.training.supervision import (
    BinaryComparison,
    SupervisionDataset,
    save_supervision_artifact_bundle,
)

if TYPE_CHECKING:
    from treepo._research.training.judges import GenRMJudge
    from treepo._research.training.judges.genrm_dspy import GenRMComparisonModule
    from treepo._research.training.generator_trainers import BaseGeneratorTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedTrainerConfig:
    """Configuration for unified training loop."""

    # Iteration limits
    max_iterations: int = 5
    min_iterations: int = 1

    # Convergence criteria
    judge_accuracy_threshold: float = 0.85
    """Target judge accuracy to consider converged."""

    violation_rate_threshold: float = 0.05
    """Target violation rate for audit-based convergence."""

    improvement_threshold: float = 0.01
    """Minimum improvement to continue (early stopping if less)."""

    patience: int = 2
    """Iterations without improvement before stopping."""

    # Preference collection
    min_preferences_for_training: int = 50
    """Minimum preferences required to trigger generator training."""

    collect_audit_preferences: bool = True
    """Also collect preferences from audit violations."""

    # Tournament configuration
    k_candidates: int = 4
    """Number of candidates per tournament."""

    n_samples_per_iteration: int = 50
    """Number of samples to process per iteration."""

    # Judge optimization
    judge_budget: str = 'medium'
    """GEPA budget for judge optimization ('light', 'medium', 'heavy')."""

    # Oracle comparison
    tie_margin: float = 0.05
    """Error margin for oracle ties."""

    normalize_errors: bool = True
    """Normalize oracle errors to 0-1."""

    scale_range: Optional[float] = None
    """Scale range for error normalization."""

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1


@dataclass
class UnifiedIterationResult:
    """Result from a single iteration of the unified training loop."""

    iteration: int
    """Iteration number (1-indexed)."""

    # Preference collection
    n_preferences_collected: int
    n_preferences_from_tournament: int
    n_preferences_from_audit: int

    # Generator training
    generator_training_completed: bool
    generator_model_path: Optional[str]
    generator_training_method: str

    # Judge optimization
    judge_accuracy_before: float
    judge_accuracy_after: float
    judge_improvement: float

    # Timing
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedTrainingResult:
    """Complete result from the unified training loop."""

    converged: bool
    convergence_reason: str  # 'accuracy_threshold', 'patience', 'max_iterations'
    final_iteration: int

    # Final metrics
    final_judge_accuracy: float
    final_generator_path: Optional[str]

    # History
    iterations: List[UnifiedIterationResult] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)

    # Paths
    optimized_judge_path: Optional[Path] = None
    final_supervision_path: Optional[Path] = None
    final_binary_projection_path: Optional[Path] = None


# =============================================================================
# Unified Trainer
# =============================================================================

class UnifiedTrainer:
    """
    Closed-loop training for judge and generator co-optimization.

    This trainer orchestrates the full training loop:
    1. Build trees with current judge → collect preferences (FREE byproduct)
    2. Enrich preferences with oracle scores
    3. Train generator using pluggable method
    4. Optimize judge via GEPA
    5. Repeat until convergence

    Both the generator AND judge are updated each iteration, creating
    a co-optimization loop where both improve together.
    """

    def __init__(
        self,
        generator_trainer: "BaseGeneratorTrainer",
        genrm_judge: "GenRMJudge",
        oracle_predict: Callable[[str], float],
        config: UnifiedTrainerConfig,
        output_dir: Union[str, Path],
        summarizer: Optional[Callable[[str, str], str]] = None,
        prompt_lm: Optional[Any] = None,
    ):
        """
        Initialize the unified trainer.

        Args:
            generator_trainer: Pluggable trainer for generator (DPO, SFT, GRPO, etc.)
            genrm_judge: GenRM judge instance (updated via GEPA each iteration)
            oracle_predict: Function(text) -> score for downstream task
            config: Training configuration
            output_dir: Directory for outputs and checkpoints
            summarizer: Optional summarizer function (for tree building)
            prompt_lm: Optional LM for DSPy prompt optimization
        """
        self.generator_trainer = generator_trainer
        self.judge = genrm_judge
        self.oracle_predict = oracle_predict
        self.config = config
        self.output_dir = Path(output_dir)
        self.summarizer = summarizer
        self.prompt_lm = prompt_lm

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Track current state
        self._current_dspy_judge: Optional['GenRMComparisonModule'] = None
        self._current_generator_path: Optional[str] = None
        self._all_supervision_dataset = SupervisionDataset()
        self._iteration_history: List[UnifiedIterationResult] = []

    def train(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
    ) -> UnifiedTrainingResult:
        """
        Run the full unified training loop.

        Args:
            samples: List of dicts with 'text', 'doc_id', 'reference_score'
            rubric: Information preservation criteria

        Returns:
            UnifiedTrainingResult with training statistics and final paths
        """
        from treepo._research.training.judges.genrm_dspy import GenRMComparisonModule

        logger.info("=" * 60)
        logger.info("UNIFIED TRAINING LOOP - Starting")
        logger.info(f"  Generator method: {self.generator_trainer.method_name}")
        logger.info(f"  Max iterations: {self.config.max_iterations}")
        logger.info(f"  Samples: {len(samples)}")
        logger.info("=" * 60)

        # Initialize DSPy-wrapped judge
        self._current_dspy_judge = GenRMComparisonModule(
            genrm_judge=self.judge,
            use_dspy_prompt=True,
            prompt_lm=self.prompt_lm,
        )

        best_accuracy = 0.0
        patience_counter = 0
        convergence_reason = 'max_iterations'

        for iteration in range(1, self.config.max_iterations + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"{'='*60}")

            result = self._run_iteration(samples, rubric, iteration)
            self._iteration_history.append(result)

            # Check convergence
            if result.judge_accuracy_after >= self.config.judge_accuracy_threshold:
                convergence_reason = 'accuracy_threshold'
                logger.info(
                    f"Converged: accuracy {result.judge_accuracy_after:.3f} >= "
                    f"{self.config.judge_accuracy_threshold}"
                )
                break

            if result.judge_accuracy_after > best_accuracy + self.config.improvement_threshold:
                best_accuracy = result.judge_accuracy_after
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"  No improvement (patience: {patience_counter}/{self.config.patience})"
                )

            if patience_counter >= self.config.patience:
                convergence_reason = 'patience'
                logger.info(f"Converged after {iteration} iterations (patience exhausted)")
                break

        # Save final artifacts
        final_judge_path = self._save_final_judge()
        final_supervision_path, final_binary_projection_path = self._save_all_supervision()

        return UnifiedTrainingResult(
            converged=convergence_reason in ('accuracy_threshold', 'patience'),
            convergence_reason=convergence_reason,
            final_iteration=len(self._iteration_history),
            final_judge_accuracy=self._iteration_history[-1].judge_accuracy_after if self._iteration_history else 0.0,
            final_generator_path=self._current_generator_path,
            iterations=self._iteration_history,
            accuracy_history=[it.judge_accuracy_after for it in self._iteration_history],
            optimized_judge_path=final_judge_path,
            final_supervision_path=final_supervision_path,
            final_binary_projection_path=final_binary_projection_path,
        )

    def _run_iteration(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
        iteration: int,
    ) -> UnifiedIterationResult:
        """Run a single iteration of the training loop."""
        iteration_start = time.time()

        # Step 1: Build trees and collect tournament preferences
        logger.info("  [1/4] Collecting tournament preferences...")
        tournament_prefs = self._collect_tournament_preferences(samples, rubric, iteration)
        logger.info(f"        Collected {len(tournament_prefs)} tournament preferences")

        # Step 2: Optionally collect audit preferences
        audit_prefs = []
        if self.config.collect_audit_preferences:
            logger.info("  [2/4] Collecting audit preferences...")
            audit_prefs = self._collect_audit_preferences(samples, rubric)
            logger.info(f"        Collected {len(audit_prefs)} audit preferences")

        # Merge preferences
        all_iteration_prefs = tournament_prefs + audit_prefs
        iteration_supervision = SupervisionDataset(
            comparative_judgments=[pref.to_comparative_judgment() for pref in all_iteration_prefs]
        )
        self._all_supervision_dataset.add_comparative_judgments(
            list(iteration_supervision.comparative_judgments)
        )

        # Step 3: Train generator
        logger.info("  [3/4] Training generator...")
        generator_trained = False
        generator_path = None

        if len(iteration_supervision.project_binary(projection="adjacent").comparisons) >= self.config.min_preferences_for_training:
            try:
                generator_path = self.generator_trainer.train(
                    preferences=iteration_supervision,
                    model_name=self._get_generator_model_name(),
                    output_dir=self.output_dir / f"generator_iter{iteration}",
                )
                generator_trained = True
                self._current_generator_path = generator_path
                logger.info(f"        Generator trained: {generator_path}")
            except Exception as e:
                logger.error(f"        Generator training failed: {e}")
        else:
            logger.warning(
                f"        Skipping generator training: "
                f"{len(iteration_supervision.project_binary(projection='adjacent').comparisons)} "
                f"< {self.config.min_preferences_for_training} preferences"
            )

        # Step 4: Enrich with oracle and optimize judge
        logger.info("  [4/4] Optimizing judge...")
        enriched = self._enrich_with_oracle(iteration_supervision, samples)
        accuracy_before = self._evaluate_judge(enriched)

        if len(enriched) >= 10:
            optimized_judge, _ = self._optimize_judge(enriched, iteration)
            self._current_dspy_judge = optimized_judge
            accuracy_after = self._evaluate_judge(enriched, optimized_judge)
        else:
            logger.warning("        Insufficient enriched preferences for judge optimization")
            accuracy_after = accuracy_before

        improvement = accuracy_after - accuracy_before
        logger.info(f"        Judge accuracy: {accuracy_before:.3f} → {accuracy_after:.3f} ({improvement:+.3f})")

        duration = time.time() - iteration_start

        return UnifiedIterationResult(
            iteration=iteration,
            n_preferences_collected=len(all_iteration_prefs),
            n_preferences_from_tournament=len(tournament_prefs),
            n_preferences_from_audit=len(audit_prefs),
            generator_training_completed=generator_trained,
            generator_model_path=generator_path,
            generator_training_method=self.generator_trainer.method_name,
            judge_accuracy_before=accuracy_before,
            judge_accuracy_after=accuracy_after,
            judge_improvement=improvement,
            duration_seconds=duration,
        )

    def _collect_tournament_preferences(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
        iteration: int,
    ) -> List[BinaryComparison]:
        """Build trees and collect binary optimizer projections from tournaments."""
        from treepo._research.tree.builder import TreeBuilder, BuildConfig
        from treepo._research.core.strategy import CallableStrategy, TournamentStrategy, TournamentConfig

        # Use current summarizer or default
        if self.summarizer is not None:
            base_strategy = CallableStrategy(self.summarizer)
        else:
            # Try to load from current generator path if available
            base_strategy = self._get_default_strategy()

        judge = self._current_dspy_judge or self.judge
        strategy = TournamentStrategy(
            base=base_strategy,
            judge=judge,
            config=TournamentConfig(
                k=self.config.k_candidates,
                temperature=0.9,
            ),
        )
        builder = TreeBuilder(strategy=strategy, config=BuildConfig())

        all_preferences = []
        samples_to_process = samples[:self.config.n_samples_per_iteration]

        for idx, sample in enumerate(samples_to_process):
            try:
                text = sample.get('text', '')
                if not text:
                    continue

                result = builder.build_sync(text, rubric)

                binary_projection = result.supervision.project_binary(projection="adjacent")
                doc_id = sample.get('doc_id', f"doc_{idx}")
                for pref in binary_projection.comparisons:
                    pref.source_example_id = doc_id
                    pref.reference_score = sample.get('reference_score')

                all_preferences.extend(binary_projection.comparisons)
                builder.reset()

            except Exception as e:
                logger.warning(f"Tree building failed for sample {idx}: {e}")

        return all_preferences

    def _collect_audit_preferences(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
    ) -> List[BinaryComparison]:
        """Collect binary optimizer projections from audit violations."""
        # This integrates with the audit system to find violations
        # and generate targeted preference pairs for them
        # For now, return empty - can be extended later
        return []

    def _enrich_with_oracle(
        self,
        supervision: SupervisionDataset,
        samples: List[Dict[str, Any]],
    ) -> SupervisionDataset:
        """Add oracle scores to supervision and return an enriched supervision dataset."""
        gt_lookup = {
            s.get('doc_id', f"doc_{i}"): s.get('reference_score')
            for i, s in enumerate(samples)
        }

        enriched = []
        for pref in supervision.project_binary(projection="adjacent").comparisons:
            try:
                score_a = self.oracle_predict(pref.summary_a)
                score_b = self.oracle_predict(pref.summary_b)
                gt = gt_lookup.get(pref.source_example_id)

                error_a = abs(score_a - gt) if gt is not None else None
                error_b = abs(score_b - gt) if gt is not None else None

                if self.config.normalize_errors and gt is not None:
                    scale_range = self.config.scale_range or 1.0
                    error_a = min(1.0, max(0.0, error_a / scale_range))
                    error_b = min(1.0, max(0.0, error_b / scale_range))

                pref.score_estimate_a = score_a
                pref.score_estimate_b = score_b
                pref.oracle_error_a = error_a
                pref.oracle_error_b = error_b
                if gt is not None:
                    pref.reference_score = gt

                enriched.append(pref)

            except Exception as e:
                logger.debug(f"Oracle enrichment failed: {e}")

        return SupervisionDataset(
            comparative_judgments=[pref.to_comparative_judgment() for pref in enriched]
        )

    def _optimize_judge(
        self,
        preferences,
        iteration: int,
    ) -> tuple['GenRMComparisonModule', dict]:
        """Train judge to predict oracle preferences using GEPA."""
        from treepo._research.training.judge_optimization import JudgeOptimizer, JudgeOptimizationConfig

        config = JudgeOptimizationConfig(
            budget=self.config.judge_budget,
            num_threads=4,
            tie_margin=self.config.tie_margin,
            checkpoint_dir=self.checkpoint_dir,
        )

        optimizer = JudgeOptimizer(config=config)

        if self.prompt_lm is not None:
            import dspy
            with dspy.context(lm=self.prompt_lm):
                optimized, results = optimizer.optimize(
                    preferences,
                    initial_judge=self._current_dspy_judge,
                )
        else:
            optimized, results = optimizer.optimize(
                preferences,
                initial_judge=self._current_dspy_judge,
            )

        # Save checkpoint
        if self.config.save_checkpoints and iteration % self.config.checkpoint_frequency == 0:
            checkpoint_path = self.checkpoint_dir / f'judge_iter_{iteration}.json'
            try:
                optimized.save(str(checkpoint_path))
            except Exception as e:
                logger.warning(f"Failed to save judge checkpoint: {e}")

        return optimized, results

    def _evaluate_judge(
        self,
        supervision: SupervisionDataset,
        judge: Optional['GenRMComparisonModule'] = None,
    ) -> float:
        """Evaluate judge accuracy on oracle-labeled supervision."""
        from treepo._research.training.judge_optimization import derive_ground_truth_preference

        judge = judge or self._current_dspy_judge
        if judge is None:
            return 0.0

        correct = 0
        total = 0

        for pref in supervision.project_binary(projection="adjacent").comparisons:
            try:
                gt = derive_ground_truth_preference(pref, tie_margin=self.config.tie_margin)
                if gt is None:
                    continue

                result = judge.forward(
                    context=pref.rubric,
                    original_text=pref.original_text,
                    summary_a=pref.summary_a,
                    summary_b=pref.summary_b,
                    law_type=pref.law_type,
                )

                predicted = getattr(result, 'preference', 'tie')
                total += 1
                if predicted == gt:
                    correct += 1

            except Exception as e:
                logger.debug(f"Evaluation error: {e}")
                total += 1

        return correct / max(1, total)

    def _get_generator_model_name(self) -> str:
        """Get the model name for generator training."""
        # Use previously trained model if available, else default
        if self._current_generator_path:
            return self._current_generator_path
        return "nvidia/Nemotron-Nano-8B"

    def _get_default_strategy(self):
        """Get a default summarization strategy."""
        from treepo._research.core.strategy import CallableStrategy

        # Simple passthrough if no summarizer configured
        def passthrough(text: str, rubric: str) -> str:
            return text[:1000]  # Truncate as simple baseline

        return CallableStrategy(passthrough)

    def _save_final_judge(self) -> Optional[Path]:
        """Save the final optimized judge."""
        if self._current_dspy_judge is None:
            return None

        judge_path = self.output_dir / 'optimized_judge' / 'judge_final.json'
        judge_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._current_dspy_judge.save(str(judge_path))
            logger.info(f"Saved final judge to {judge_path}")
            return judge_path
        except Exception as e:
            logger.warning(f"Failed to save final judge: {e}")
            return None

    def _save_all_supervision(self) -> tuple[Optional[Path], Optional[Path]]:
        """Save all collected supervision and an explicit binary optimizer export."""
        if len(self._all_supervision_dataset) == 0:
            return None, None

        supervision_path = self.output_dir / 'all_supervision.json'
        binary_projection_path = self.output_dir / 'all_binary_projection.json'
        try:
            save_supervision_artifact_bundle(
                self._all_supervision_dataset,
                supervision_path=supervision_path,
            )
            logger.info("Saved collected supervision to %s", supervision_path)
            if len(self._all_supervision_dataset.project_binary(projection="adjacent")) > 0:
                self._all_supervision_dataset.project_binary(
                    projection="adjacent"
                ).save(binary_projection_path)
                return supervision_path, binary_projection_path
            return supervision_path, None
        except Exception as e:
            logger.warning(f"Failed to save collected supervision: {e}")
            return None, None

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def current_judge(self) -> Optional['GenRMComparisonModule']:
        """Get the current optimized judge."""
        return self._current_dspy_judge

    @property
    def current_generator_path(self) -> Optional[str]:
        """Get the path to the current trained generator."""
        return self._current_generator_path

    @property
    def all_binary_projection(self) -> List[BinaryComparison]:
        """Get all collected binary optimizer projections."""
        return list(self._all_supervision_dataset.project_binary(projection="adjacent").comparisons)

    @property
    def all_supervision_dataset(self) -> SupervisionDataset:
        """Get all collected supervision as the primary dataset surface."""
        return self._all_supervision_dataset

    @property
    def history(self) -> List[UnifiedIterationResult]:
        """Get iteration history."""
        return self._iteration_history.copy()


# =============================================================================
# Convenience Functions
# =============================================================================

def run_unified_training(
    generator_method: Literal["dpo", "sft", "grpo", "bootstrap_finetune"],
    genrm_judge: "GenRMJudge",
    oracle_predict: Callable[[str], float],
    samples: List[Dict[str, Any]],
    rubric: str,
    output_dir: Union[str, Path],
    max_iterations: int = 5,
    summarizer: Optional[Callable[[str, str], str]] = None,
    **trainer_kwargs,
) -> UnifiedTrainingResult:
    """
    Convenience function to run unified training.

    Args:
        generator_method: Training method for generator
        genrm_judge: GenRM judge instance
        oracle_predict: Oracle scoring function
        samples: Training samples
        rubric: Information preservation criteria
        output_dir: Output directory
        max_iterations: Maximum training iterations
        summarizer: Optional summarizer function
        **trainer_kwargs: Additional arguments for generator trainer

    Returns:
        UnifiedTrainingResult
    """
    from treepo._research.training.generator_trainers import create_trainer_from_method

    generator_trainer = create_trainer_from_method(
        method=generator_method,
        genrm_judge=genrm_judge,
        **trainer_kwargs,
    )

    config = UnifiedTrainerConfig(max_iterations=max_iterations)

    trainer = UnifiedTrainer(
        generator_trainer=generator_trainer,
        genrm_judge=genrm_judge,
        oracle_predict=oracle_predict,
        config=config,
        output_dir=output_dir,
        summarizer=summarizer,
    )

    return trainer.train(samples, rubric)
