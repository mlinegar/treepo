"""
Tournament of Tournaments - Iterative Judge Optimization.

This module implements the full iterative tournament of tournaments loop for
improving the GenRM judge's ability to predict which summaries lead to better
downstream oracle performance.

The loop:
1. Build trees with current judge → collect preferences as free byproduct
2. Enrich preferences with oracle scores (ground truth from downstream task)
3. Optimize judge to predict oracle preferences
4. Evaluate improvement and check convergence
5. Repeat until convergence

The key insight is that the training signal comes from **downstream oracle
performance**, not from the judge's own scores. This avoids circular logic.

Usage:
    from treepo._research.training.tournament_loop import (
        TournamentOfTournamentsTrainer,
        ToTConfig,
    )

    # Create trainer
    trainer = TournamentOfTournamentsTrainer(
        summarizer=summarizer,
        oracle_predict=task.create_oracle_scorer(),
        initial_judge=judge,
        config=ToTConfig(max_iterations=5),
        output_dir=output_dir,
    )

    # Run training loop
    result = trainer.train(samples, rubric)
    print(f"Final accuracy: {result.final_judge_accuracy:.3f}")
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from treepo._research.training.supervision import BinaryComparison, ComparativeJudgment, SupervisionDataset

if TYPE_CHECKING:
    from treepo._research.training.judges import GenRMJudge
    from treepo._research.training.judges.genrm_dspy import GenRMComparisonModule

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ToTConfig:
    """Configuration for Tournament of Tournaments."""

    # Iteration limits
    max_iterations: int = 5
    min_iterations: int = 1

    # Convergence criteria
    convergence_threshold: float = 0.01  # Stop if improvement < this
    convergence_patience: int = 2        # Stop after N iterations without improvement

    # Tree building
    # Number of candidate summaries generated per tournament round.
    # 4 provides a good diversity-cost tradeoff:
    # - Higher (8+): More diversity but 4x more LLM calls per merge
    # - Lower (2): Faster but less exploration of summary space
    # - 4: 4 candidates → 6 pairwise comparisons, covers space well
    k_candidates: int = 4
    n_samples_per_iteration: int = 50    # Samples to process per iteration
    candidate_temperature: float = 0.9   # Temperature for candidate generation

    # Judge optimization
    judge_budget: str = 'medium'         # 'light', 'medium', 'heavy', 'superheavy'
    num_threads: int = 4                 # Parallel evaluation threads
    judge_test_split: float = 0.2        # Holdout split for judge evaluation

    # Oracle comparison (normalized units when errors are normalized)
    tie_margin: float = 0.05             # Error difference below this = tie
    normalize_errors: bool = True        # Normalize errors to 0-1
    scale_range: Optional[float] = None  # Range for normalization (defaults to 1.0)
                                         # Use the task's scale range when available

    # Sampling
    shuffle_samples_each_iteration: bool = True
    random_seed: int = 42

    # Preference labeling (optional override)
    preference_labeler: Optional[Callable[[BinaryComparison, float], Optional[str]]] = None

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1        # Save every N iterations


@dataclass
class ToTIterationResult:
    """Result from one iteration of the training loop."""

    iteration: int
    n_trees_built: int
    n_binary_projection_records: int
    n_comparative_judgments_collected: int
    n_binary_records_with_oracle: int
    judge_accuracy_before: float
    judge_accuracy_after: float
    improvement: float
    duration_seconds: float


@dataclass
class ToTResult:
    """Complete training result from the tournament of tournaments loop."""

    converged: bool
    convergence_reason: str  # 'patience', 'threshold', 'max_iterations'
    final_iteration: int
    iterations: List[ToTIterationResult] = field(default_factory=list)
    final_judge_accuracy: float = 0.0
    improvement_history: List[float] = field(default_factory=list)
    optimized_judge_path: Optional[Path] = None


# =============================================================================
# Tournament of Tournaments Trainer
# =============================================================================

class TournamentOfTournamentsTrainer:
    """
    Full iterative tournament of tournaments loop.

    Each iteration:
    1. Build trees with current judge → collect preferences
    2. Enrich preferences with oracle scores
    3. Optimize judge to predict oracle preferences
    4. Evaluate improvement
    5. Check convergence

    The training signal comes from downstream oracle performance,
    not from the judge's own scores. This is key to avoiding circular logic.
    """

    def __init__(
        self,
        summarizer: Callable[[str, str], str],
        oracle_predict: Callable[[str], float],
        initial_judge: 'GenRMJudge',
        config: ToTConfig,
        output_dir: Path,
        prompt_lm: Optional[Any] = None,
    ):
        """
        Initialize the trainer.

        Args:
            summarizer: Function(content, rubric) -> summary
            oracle_predict: Function(text) -> score
            initial_judge: GenRMJudge instance for initial tournament selection
            config: Training configuration
            output_dir: Directory for outputs and checkpoints
        """
        self.summarizer = summarizer
        self.oracle_predict = oracle_predict
        self.judge = initial_judge
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_lm = prompt_lm

        # Create checkpoint directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Track current DSPy-wrapped judge (for optimization)
        self._current_dspy_judge: Optional['GenRMComparisonModule'] = None

        # Track all collected supervision for downstream export.
        self._all_supervision_dataset = SupervisionDataset()

    def train(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
    ) -> ToTResult:
        """
        Run full tournament of tournaments loop.

        Args:
            samples: List of dicts with 'text', 'doc_id', 'reference_score'
            rubric: Information preservation criteria

        Returns:
            ToTResult with training statistics and final judge path
        """
        from treepo._research.training.judges.genrm_dspy import GenRMComparisonModule

        # Initialize prompt-tuned GenRM judge for optimization + tournament selection
        self._current_dspy_judge = GenRMComparisonModule(
            genrm_judge=self.judge,
            use_dspy_prompt=True,
            prompt_lm=self.prompt_lm,
        )

        iterations = []
        best_accuracy = 0.0
        patience_counter = 0
        convergence_reason = 'max_iterations'
        final_optimized_judge = None

        for iteration in range(1, self.config.max_iterations + 1):
            iteration_start = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"TOURNAMENT OF TOURNAMENTS - Iteration {iteration}")
            logger.info(f"{'='*60}")

            # Step 1: Build trees with current judge, collect preferences
            supervision_dataset = self._build_trees_and_collect_supervision(
                samples, rubric, iteration
            )

            if (
                supervision_dataset is None
                or (
                    len(supervision_dataset) == 0
                    and len(getattr(supervision_dataset, "comparative_judgments", []) or []) == 0
                )
            ):
                logger.warning(f"Iteration {iteration}: No supervision collected, skipping")
                continue

            # Accumulate supervision for export.
            self._all_supervision_dataset.add_comparative_judgments(
                list(getattr(supervision_dataset, "comparative_judgments", []) or [])
            )

            # Step 2: Enrich with oracle scores
            enriched_dataset = self._enrich_with_oracle(supervision_dataset, samples)
            optimizer_dataset = enriched_dataset

            optimizer_pair_count = len(
                optimizer_dataset.project_binary(projection="adjacent").comparisons
            )
            if optimizer_pair_count < 10:
                logger.warning(
                    "Iteration %d: Only %d optimizer-ready binary records after oracle enrichment, may be insufficient",
                    iteration,
                    optimizer_pair_count,
                )

            # Step 3: Optimize judge and evaluate on holdout
            optimized_judge, opt_results = self._optimize_judge(optimizer_dataset, iteration)
            final_optimized_judge = optimized_judge

            if opt_results and 'baseline' in opt_results and 'optimized' in opt_results:
                accuracy_before = opt_results['baseline'].get('accuracy', 0.0)
                accuracy_after = opt_results['optimized'].get('accuracy', 0.0)
            else:
                accuracy_before = self._evaluate_judge(optimizer_dataset)
                accuracy_after = self._evaluate_judge(optimizer_dataset, optimized_judge)

            logger.info(f"  Judge accuracy before: {accuracy_before:.3f}")
            logger.info(f"  Judge accuracy after: {accuracy_after:.3f}")

            improvement = accuracy_after - accuracy_before
            logger.info(f"  Improvement: {improvement:+.3f}")

            iteration_duration = time.time() - iteration_start

            # Record iteration result
            result = ToTIterationResult(
                iteration=iteration,
                n_trees_built=len(samples[:self.config.n_samples_per_iteration]),
                n_binary_projection_records=len(
                    supervision_dataset.project_binary(projection="adjacent").comparisons
                ),
                n_comparative_judgments_collected=len(
                    getattr(supervision_dataset, "comparative_judgments", []) or []
                ),
                n_binary_records_with_oracle=optimizer_pair_count,
                judge_accuracy_before=accuracy_before,
                judge_accuracy_after=accuracy_after,
                improvement=improvement,
                duration_seconds=iteration_duration,
            )
            iterations.append(result)

            # Step 4: Update internal state for next iteration
            self._current_dspy_judge = optimized_judge

            # Step 5: Check convergence
            if accuracy_after > best_accuracy + self.config.convergence_threshold:
                best_accuracy = accuracy_after
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"  No improvement (patience: {patience_counter}/{self.config.convergence_patience})")

            if patience_counter >= self.config.convergence_patience:
                convergence_reason = 'patience'
                logger.info(f"Converged after {iteration} iterations (patience exhausted)")
                break

            if iteration >= self.config.min_iterations and improvement < self.config.convergence_threshold:
                convergence_reason = 'threshold'
                logger.info(f"Converged after {iteration} iterations (minimal improvement)")
                break

            # Save checkpoint
            if self.config.save_checkpoints and iteration % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(optimized_judge, iteration)

        # Save final judge
        judge_path = self.output_dir / 'optimized_judge' / 'judge_final.json'
        if final_optimized_judge is not None:
            self._save_judge(final_optimized_judge, judge_path)
        else:
            judge_path = None

        return ToTResult(
            converged=convergence_reason in ('patience', 'threshold'),
            convergence_reason=convergence_reason,
            final_iteration=iteration if iterations else 0,
            iterations=iterations,
            final_judge_accuracy=accuracy_after if iterations else 0.0,
            improvement_history=[it.improvement for it in iterations],
            optimized_judge_path=judge_path,
        )

    def _build_trees_and_collect_supervision(
        self,
        samples: List[Dict[str, Any]],
        rubric: str,
        iteration: int,
    ):
        """
        Build trees using current judge and return the collected supervision dataset.

        Binary optimizer projections remain derivable on demand, but the primary
        collected object is a supervision dataset with comparative judgments.
        """
        from treepo._research.tree.builder import TreeBuilder, BuildConfig
        from treepo._research.core.strategy import (
            CallableStrategy,
            TournamentStrategy,
            TournamentConfig,
            tournament_doc_id,
        )
        base_strategy = CallableStrategy(self.summarizer)
        judge = self._current_dspy_judge or self.judge
        strategy = TournamentStrategy(
            base=base_strategy,
            judge=judge,
            config=TournamentConfig(
                k=self.config.k_candidates,
                temperature=self.config.candidate_temperature,
            ),
        )
        builder = TreeBuilder(strategy=strategy, config=BuildConfig())

        collected = SupervisionDataset()
        samples_to_process = list(samples)
        if self.config.shuffle_samples_each_iteration:
            rng = random.Random(self.config.random_seed + iteration)
            rng.shuffle(samples_to_process)
        samples_to_process = samples_to_process[:self.config.n_samples_per_iteration]

        logger.info(f"  Building {len(samples_to_process)} trees...")

        for idx, sample in enumerate(samples_to_process):
            try:
                text = sample.get('text', '')
                if not text:
                    continue

                doc_id = sample.get('doc_id', f"doc_{idx}")
                token = tournament_doc_id.set(str(doc_id))
                try:
                    result = builder.build_sync(text, rubric)
                finally:
                    tournament_doc_id.reset(token)

                direct_binary_projection = result.supervision.project_binary(
                    projection="adjacent"
                )
                for pref in direct_binary_projection.comparisons:
                    pref.source_example_id = doc_id
                    pref.reference_score = sample.get('reference_score')
                for record in list(result.supervision.comparative_judgments):
                    record.source_example_id = doc_id
                    record.reference_score = float(sample.get('reference_score') or 0.0)

                collected.add_comparative_judgments(
                    list(result.supervision.comparative_judgments)
                )

            except Exception as e:
                logger.warning(f"  Tree building failed for sample {idx}: {e}")

            # Reset for next tree
            builder.reset()

        logger.info(
            "  Collected %d binary projections and %d comparative judgments from %d trees",
            len(collected.project_binary(projection="adjacent").comparisons),
            len(collected.comparative_judgments),
            len(samples_to_process),
        )
        return collected

    def _enrich_with_oracle(
        self,
        supervision_dataset,
        samples: List[Dict[str, Any]],
    ):
        """
        Add oracle scores to collected supervision records.

        Pairwise records keep the legacy oracle-error annotations used by the
        binary judge optimizer. Comparative records are also re-ranked by the
        same oracle signal so grouped objectives can consume the same iteration.
        """
        # Create lookup for ground truth scores (if available)
        gt_lookup = {
            s.get('doc_id', f"doc_{i}"): s.get('reference_score')
            for i, s in enumerate(samples)
        }

        enriched_records = []
        for record in list(getattr(supervision_dataset, "comparative_judgments", []) or []):
            try:
                gt = gt_lookup.get(record.source_example_id)
                candidate_rows = []
                for candidate in record.candidates:
                    oracle_score = self.oracle_predict(candidate.response)
                    oracle_error = abs(float(oracle_score) - float(gt)) if gt is not None else None
                    if self.config.normalize_errors and oracle_error is not None:
                        scale_range = self.config.scale_range
                        if scale_range is None:
                            logger.warning(
                                "No scale_range specified for error normalization. "
                                "Using default 1.0 (DSPy convention). "
                                "Set scale_range to the task scale range if normalization is needed."
                            )
                            scale_range = 1.0
                        if scale_range <= 0:
                            raise ValueError(f"scale_range must be positive, got {scale_range}.")
                        oracle_error = min(1.0, max(0.0, float(oracle_error) / float(scale_range)))
                    utility = -float(oracle_error) if oracle_error is not None else float(oracle_score)
                    candidate.metadata["oracle_score"] = float(oracle_score)
                    if oracle_error is not None:
                        candidate.metadata["oracle_error"] = float(oracle_error)
                    candidate.response_signal_value = float(utility)
                    candidate_rows.append(
                        (candidate, float(utility), oracle_error)
                    )

                candidate_rows.sort(
                    key=lambda item: (
                        -float(item[1]),
                        item[0].candidate_id,
                    )
                )
                prev_utility = None
                current_rank = 0
                for index, (candidate, utility, _oracle_error) in enumerate(candidate_rows, start=1):
                    if prev_utility is None or abs(float(utility) - float(prev_utility)) >= float(
                        self.config.tie_margin
                    ):
                        current_rank = index
                        prev_utility = float(utility)
                    candidate.rank = current_rank

                top_utility = candidate_rows[0][1] if candidate_rows else None
                next_utility = candidate_rows[1][1] if len(candidate_rows) > 1 else None
                record.preference_supervision = record.preference_supervision.with_updates(
                    response_signal_name="oracle_relative_utility",
                    comparison_signal_name="oracle_relative_margin",
                    metadata={
                        **dict(record.preference_supervision.metadata or {}),
                        "oracle_enriched": True,
                    },
                )
                record.comparison_signal_value = (
                    float(top_utility) - float(next_utility)
                    if top_utility is not None and next_utility is not None
                    else None
                )
                record.metadata["oracle_enriched"] = True
                record.metadata["oracle_tie_margin"] = float(self.config.tie_margin)
                if gt is not None:
                    record.reference_score = float(gt)
                enriched_records.append(record)
            except Exception as e:
                logger.debug(f"Oracle enrichment failed for comparative record: {e}")

        logger.info(
            "  Enriched %d projected binary records from %d/%d comparative judgments",
            len(supervision_dataset.project_binary(projection="adjacent").comparisons),
            len(enriched_records),
            len(getattr(supervision_dataset, "comparative_judgments", []) or []),
        )
        return SupervisionDataset(
            comparative_judgments=enriched_records,
        )

    def _optimize_judge(
        self,
        supervision,
        iteration: int,
    ) -> tuple['GenRMComparisonModule', dict]:
        """
        Train judge to predict oracle preferences using GEPA.

        The ground truth is derived from oracle errors:
        - Lower oracle error = better summary
        - We train the judge to predict which summary has lower oracle error
        """
        from treepo._research.training.judge_optimization import JudgeOptimizer, JudgeOptimizationConfig

        config = JudgeOptimizationConfig(
            budget=self.config.judge_budget,
            num_threads=self.config.num_threads,
            tie_margin=self.config.tie_margin,
            test_split=self.config.judge_test_split,
            checkpoint_dir=self.checkpoint_dir,
            preference_labeler=self.config.preference_labeler,
        )

        optimizer = JudgeOptimizer(config=config)
        if self.prompt_lm is not None:
            import dspy
            with dspy.context(lm=self.prompt_lm):
                optimized, results = optimizer.optimize(
                    supervision,
                    initial_judge=self._current_dspy_judge,
                )
        else:
            optimized, results = optimizer.optimize(
                supervision,
                initial_judge=self._current_dspy_judge,
            )

        logger.info(f"  Optimization results: {results}")

        return optimized, results

    def _evaluate_judge(
        self,
        supervision,
        judge: Optional['GenRMComparisonModule'] = None,
    ) -> float:
        """
        Evaluate judge accuracy on oracle-labeled preferences.

        Returns accuracy: proportion of correct preference predictions.
        """
        from treepo._research.training.judge_optimization import derive_ground_truth_preference

        judge = judge or self._current_dspy_judge
        if judge is None:
            return 0.0

        correct = 0
        total = 0

        pair_list = list(
            supervision.project_binary(projection="adjacent").comparisons
        )

        for pref in pair_list:
            try:
                # Get oracle ground truth
                gt = derive_ground_truth_preference(
                    pref,
                    tie_margin=self.config.tie_margin,
                    preference_labeler=self.config.preference_labeler,
                )
                if gt is None:
                    continue

                # Get judge prediction
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
                total += 1  # Count as error (incorrect)

        return correct / max(1, total)

    def _save_checkpoint(
        self,
        judge: 'GenRMComparisonModule',
        iteration: int,
    ) -> None:
        """Save iteration checkpoint."""
        checkpoint_path = self.checkpoint_dir / f'judge_iter_{iteration}.json'
        self._save_judge(judge, checkpoint_path)
        logger.info(f"  Saved checkpoint: {checkpoint_path}")

    def _save_judge(
        self,
        judge: 'GenRMComparisonModule',
        path: Path,
    ) -> None:
        """Save judge to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            judge.save(str(path))
        except Exception as e:
            logger.warning(f"Failed to save judge to {path}: {e}")

    # =========================================================================
    # Export Methods for Unified Training
    # =========================================================================

    def get_optimized_judge(self) -> Optional['GenRMComparisonModule']:
        """
        Return the current optimized judge for use in other pipelines.

        This method allows the tournament-optimized judge to be extracted
        and used in downstream training (e.g., unified training loop).

        Returns:
            Current optimized GenRMComparisonModule, or None if not yet optimized
        """
        return self._current_dspy_judge

    def get_tournament_winners(self) -> List[tuple]:
        """
        Return (content, rubric, winner) tuples from collected preferences.

        Extracts the winning summaries from all tournament comparisons.
        Useful for SFT training or BootstrapFinetune.

        Returns:
            List of (original_text, rubric, winning_summary) tuples
        """
        winners = []
        for pref in self._all_supervision_dataset.project_binary(projection="adjacent").comparisons:
            winner = pref.get_winner()
            if winner is not None:
                winners.append((
                    pref.original_text,
                    pref.rubric,
                    winner,
                ))
        return winners

    def get_all_binary_projection(self) -> List[BinaryComparison]:
        """Return all collected binary optimizer projections."""
        return list(self._all_supervision_dataset.project_binary(projection="adjacent").comparisons)

    def get_all_binary_projection_dataset(self):
        """Return the binary optimizer view of collected supervision."""
        return self._all_supervision_dataset.project_binary(projection="adjacent")

    def get_all_supervision_dataset(self):
        """Return the full collected supervision dataset."""
        return self._all_supervision_dataset

    def get_all_comparative_judgments(self) -> List[ComparativeJudgment]:
        """Return all collected comparative judgments from tournament iterations."""
        return list(getattr(self._all_supervision_dataset, "comparative_judgments", []) or [])


# =============================================================================
# Convenience Functions
# =============================================================================

def run_tournament_of_tournaments(
    summarizer: Callable[[str, str], str],
    oracle_predict: Callable[[str], float],
    initial_judge: 'GenRMJudge',
    samples: List[Dict[str, Any]],
    rubric: str,
    output_dir: Path,
    max_iterations: int = 5,
    k_candidates: int = 4,
    judge_budget: str = 'medium',
    prompt_lm: Optional[Any] = None,
) -> ToTResult:
    """
    Convenience function to run tournament of tournaments.

    Args:
        summarizer: Function(content, rubric) -> summary
        oracle_predict: Function(text) -> score
        initial_judge: GenRMJudge instance
        samples: List of {text, doc_id, reference_score}
        rubric: Information preservation criteria
        output_dir: Output directory
        max_iterations: Maximum iterations
        k_candidates: Candidates per tournament
        judge_budget: GEPA budget

    Returns:
        ToTResult with training statistics
    """
    config = ToTConfig(
        max_iterations=max_iterations,
        k_candidates=k_candidates,
        judge_budget=judge_budget,
    )

    trainer = TournamentOfTournamentsTrainer(
        summarizer=summarizer,
        oracle_predict=oracle_predict,
        initial_judge=initial_judge,
        config=config,
        output_dir=output_dir,
        prompt_lm=prompt_lm,
    )

    return trainer.train(samples, rubric)


# Re-export from judge_optimization for backward compatibility
from treepo._research.training.judge_optimization import load_optimized_judge
