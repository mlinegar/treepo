"""
Audit-Driven Training Pipeline.

This module connects the OPS audit system (which discovers local law violations)
to the TRL training pipeline. When audits find violations, this system:

1. Extracts failed nodes from the audit report
2. Generates preference pairs for the violated laws
3. Triggers training (DPO, GRPO, pairwise reward, or scalar reward model)
4. Re-audits to verify improvement

This implements the closed-loop improvement cycle:
    audit → discover violations → collect preferences → train → re-audit

Usage:
    from treepo._research.training.audit_driven_training import AuditDrivenTrainer

    trainer = AuditDrivenTrainer(
        model_name="nvidia/Nemotron-Nano-8B",
        judge=genrm_judge,
        output_dir="models/audit_trained",
    )

    # Run single improvement iteration
    result = trainer.run_iteration(
        audit_report=report,
        training_method="dpo",
    )

    # Run until convergence
    results = trainer.run_until_converged(
        audit_fn=lambda: auditor.run(tree),
        max_iterations=5,
        violation_threshold=0.05,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from treepo._research.training.supervision import (
    BinaryComparison,
    SupervisionDataset,
    save_supervision_artifact_bundle,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class AuditDrivenConfig:
    """Configuration for audit-driven training."""

    # Training method
    training_method: Literal["dpo", "grpo", "reward", "scalar_reward"] = "dpo"
    """Which TRL trainer to use."""

    # Preference collection
    k_candidates: int = 4
    """Number of candidates to generate per violation."""

    collect_from_violations_only: bool = True
    """Only collect preferences from violated nodes (vs. all nodes)."""

    # Training
    min_preferences_for_training: int = 50
    """Minimum preference pairs before training."""

    # Convergence
    max_iterations: int = 5
    """Maximum training iterations."""

    violation_threshold: float = 0.05
    """Target violation rate to consider converged."""

    improvement_threshold: float = 0.01
    """Minimum improvement to continue (early stopping if less)."""

    # Output
    save_intermediate: bool = True
    """Save models after each iteration."""


@dataclass
class IterationResult:
    """Result from a single training iteration."""

    iteration: int
    """Iteration number (1-indexed)."""

    violations_before: Dict[str, float]
    """Violation rates before training (by law type)."""

    violations_after: Optional[Dict[str, float]]
    """Violation rates after training (None if not re-audited)."""

    num_binary_projection_records: int
    """Number of binary optimizer records collected."""

    training_completed: bool
    """Whether training was executed."""

    model_path: Optional[str]
    """Path to saved model (if training completed)."""

    timestamp: str
    """ISO timestamp."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    @property
    def improvement(self) -> Optional[float]:
        """Total violation rate improvement (positive = better)."""
        if self.violations_after is None:
            return None
        before_total = sum(self.violations_before.values())
        after_total = sum(self.violations_after.values())
        return before_total - after_total


# =============================================================================
# Audit-Driven Trainer
# =============================================================================

class AuditDrivenTrainer:
    """
    Orchestrates the audit → train → re-audit loop.

    This class bridges the audit system with TRL training:
    1. Takes audit reports identifying violations
    2. Extracts failed nodes and generates preference data
    3. Trains model using TRL (DPO, GRPO, or reward model)
    4. Can re-audit to verify improvement
    """

    def __init__(
        self,
        model_name: str,
        judge: Any,  # GenRMJudge or similar
        summarizer: Any,  # Summarizer module
        output_dir: Union[str, Path],
        config: Optional[AuditDrivenConfig] = None,
    ):
        """
        Initialize audit-driven trainer.

        Args:
            model_name: HuggingFace model name to fine-tune
            judge: Judge for collecting preferences (GenRMJudge)
            summarizer: Summarizer module for generating candidates
            output_dir: Base output directory
            config: Training configuration
        """
        self.model_name = model_name
        self.judge = judge
        self.summarizer = summarizer
        self.output_dir = Path(output_dir)
        self.config = config or AuditDrivenConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._iteration_history: List[IterationResult] = []

    def extract_violations(
        self,
        audit_report: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract violation information from audit report.

        Args:
            audit_report: AuditReport from auditor

        Returns:
            Dict mapping law_type to list of violation details
        """
        violations = {
            "sufficiency": [],
            "idempotence": [],
            "merge": [],
        }

        # Extract from audit report structure
        # Adapt this based on actual AuditReport structure
        if hasattr(audit_report, 'failed_checks'):
            for check in audit_report.failed_checks:
                law_type = getattr(check, 'law_type', 'sufficiency')
                violations[law_type].append({
                    "node_id": getattr(check, 'node_id', None),
                    "input_text": getattr(check, 'input_text', ''),
                    "output_text": getattr(check, 'output_text', ''),
                    "rubric": getattr(check, 'rubric', ''),
                    "score": getattr(check, 'score', 0.0),
                })

        # Alternative: extract from review queue
        if hasattr(audit_report, 'review_queue'):
            for item in audit_report.review_queue:
                law_type = getattr(item, 'law_type', 'sufficiency')
                violations[law_type].append({
                    "node_id": getattr(item, 'node_id', None),
                    "input_text": getattr(item, 'input_text', ''),
                    "output_text": getattr(item, 'output_text', ''),
                    "rubric": getattr(item, 'rubric', ''),
                    "score": getattr(item, 'score', 0.0),
                })

        return violations

    def collect_preferences_for_violations(
        self,
        violations: Dict[str, List[Dict[str, Any]]],
    ) -> SupervisionDataset:
        """
        Collect supervision for violated nodes.

        Generates candidate summaries and collects preferences
        via the judge (GenRM or oracle).

        Args:
            violations: Dict from extract_violations()

        Returns:
            SupervisionDataset with collected comparative judgments
        """
        pairs: List[BinaryComparison] = []

        for law_type, violation_list in violations.items():
            logger.info(f"Collecting preferences for {len(violation_list)} {law_type} violations")

            for violation in violation_list:
                input_text = violation.get("input_text", "")
                rubric = violation.get("rubric", "")

                if not input_text:
                    continue

                try:
                    # Generate candidates
                    candidates = self._generate_candidates(input_text, rubric)

                    if len(candidates) < 2:
                        continue

                    # Collect pairwise preferences
                    for i in range(len(candidates)):
                        for j in range(i + 1, len(candidates)):
                            pair = self._compare_candidates(
                                original_text=input_text,
                                rubric=rubric,
                                summary_a=candidates[i],
                                summary_b=candidates[j],
                                law_type=law_type,
                            )
                            if pair:
                                pairs.append(pair)

                except Exception as e:
                    logger.warning(f"Failed to collect preferences for violation: {e}")

        logger.info(f"Collected {len(pairs)} binary supervision records total")
        return SupervisionDataset(
            comparative_judgments=[pair.to_comparative_judgment() for pair in pairs]
        )

    def _generate_candidates(
        self,
        input_text: str,
        rubric: str,
    ) -> List[str]:
        """Generate candidate summaries using the summarizer."""
        candidates = []
        temperatures = [0.3, 0.5, 0.7, 0.9][:self.config.k_candidates]

        for temp in temperatures:
            try:
                # Adapt this based on actual summarizer interface
                if hasattr(self.summarizer, '__call__'):
                    result = self.summarizer(input_text, rubric)
                    if hasattr(result, 'summary'):
                        candidates.append(result.summary)
                    elif isinstance(result, str):
                        candidates.append(result)
            except Exception as e:
                logger.warning(f"Candidate generation failed at temp={temp}: {e}")

        return candidates

    def _compare_candidates(
        self,
        original_text: str,
        rubric: str,
        summary_a: str,
        summary_b: str,
        law_type: str,
    ) -> Optional[BinaryComparison]:
        """Compare two candidates using the judge."""
        from treepo._research.training.supervision import BinaryComparison
        import uuid
        from treepo._research.core.supervision_metadata import judgment_supervision_metadata

        try:
            result = self.judge.compare(
                context=rubric,
                original_text=original_text,
                summary_a=summary_a,
                summary_b=summary_b,
            )

            return BinaryComparison(
                pair_id=str(uuid.uuid4()),
                source_example_id=f"audit_{law_type}",
                original_text=original_text,
                rubric=rubric,
                reference_score=0.0,  # Unknown for audit violations
                summary_a=summary_a,
                summary_b=summary_b,
                preferred=result.preferred,
                reasoning=getattr(result, 'reasoning', ''),
                confidence=getattr(result, 'confidence', 0.5),
                law_type=law_type,
                preference_supervision=judgment_supervision_metadata(
                    application_name="audit_driven_training",
                    law_type=law_type,
                ),
                score_estimate_a=getattr(result, 'helpfulness_a', None),
                score_estimate_b=getattr(result, 'helpfulness_b', None),
            )

        except Exception as e:
            logger.warning(f"Comparison failed: {e}")
            return None

    def run_iteration(
        self,
        audit_report: Any,
        training_method: Optional[str] = None,
        re_audit_fn: Optional[Callable[[], Any]] = None,
    ) -> IterationResult:
        """
        Run a single training iteration.

        Args:
            audit_report: AuditReport from auditor
            training_method: Override training method from config
            re_audit_fn: Function to re-audit after training (returns new report)

        Returns:
            IterationResult with details
        """
        iteration_num = len(self._iteration_history) + 1
        method = training_method or self.config.training_method

        logger.info(f"=== Iteration {iteration_num}: {method.upper()} Training ===")

        # Extract violation rates before training
        violations_before = self._compute_violation_rates(audit_report)
        logger.info(f"Violation rates before: {violations_before}")

        # Extract and collect preferences
        violations = self.extract_violations(audit_report)
        total_violations = sum(len(v) for v in violations.values())

        if total_violations == 0:
            logger.info("No violations found, skipping training")
            result = IterationResult(
                iteration=iteration_num,
                violations_before=violations_before,
                violations_after=None,
                num_binary_projection_records=0,
                training_completed=False,
                model_path=None,
                timestamp=datetime.now().isoformat(),
                metadata={"reason": "no_violations"},
            )
            self._iteration_history.append(result)
            return result

        supervision = self.collect_preferences_for_violations(violations)

        projected_pairs = supervision.project_binary(projection="adjacent").comparisons
        if len(projected_pairs) < self.config.min_preferences_for_training:
            logger.warning(
                f"Insufficient preferences ({len(projected_pairs)} < "
                f"{self.config.min_preferences_for_training}), skipping training"
            )
            result = IterationResult(
                iteration=iteration_num,
                violations_before=violations_before,
                violations_after=None,
                num_binary_projection_records=len(projected_pairs),
                training_completed=False,
                model_path=None,
                timestamp=datetime.now().isoformat(),
                metadata={"reason": "insufficient_binary_projection"},
            )
            self._iteration_history.append(result)
            return result

        # Save primary supervision. Binary projections are emitted only when needed.
        supervision_path = self.output_dir / f"supervision_iter{iteration_num}.json"
        save_supervision_artifact_bundle(
            supervision,
            supervision_path=supervision_path,
        )

        # Train model
        from treepo._research.training.trl_training import (
            TRLTrainingConfig,
            train_dpo,
            train_grpo,
            train_reward_model,
            train_scalar_reward_model,
        )

        model_dir = self.output_dir / f"model_iter{iteration_num}"
        trl_config = TRLTrainingConfig()

        if method == "dpo":
            model_path = train_dpo(supervision, self.model_name, model_dir, trl_config)
        elif method == "grpo":
            model_path = train_grpo(supervision, self.model_name, model_dir, trl_config)
        elif method == "reward":
            model_path = train_reward_model(supervision, self.model_name, model_dir, trl_config)
        elif method == "scalar_reward":
            model_path = train_scalar_reward_model(
                supervision,
                self.model_name,
                model_dir,
                trl_config,
            )
        else:
            raise ValueError(f"Unknown training method: {method}")

        # Re-audit if function provided
        violations_after = None
        if re_audit_fn:
            logger.info("Re-auditing after training...")
            new_report = re_audit_fn()
            violations_after = self._compute_violation_rates(new_report)
            logger.info(f"Violation rates after: {violations_after}")

        result = IterationResult(
            iteration=iteration_num,
            violations_before=violations_before,
            violations_after=violations_after,
            num_binary_projection_records=len(projected_pairs),
            training_completed=True,
            model_path=model_path,
            timestamp=datetime.now().isoformat(),
            metadata={
                "method": method,
                "supervision_path": str(supervision_path),
            },
        )
        self._iteration_history.append(result)

        return result

    def run_until_converged(
        self,
        audit_fn: Callable[[], Any],
        training_method: Optional[str] = None,
        max_iterations: Optional[int] = None,
        violation_threshold: Optional[float] = None,
    ) -> List[IterationResult]:
        """
        Run training iterations until convergence.

        Convergence is defined as:
        - Violation rate below threshold, OR
        - No improvement between iterations, OR
        - Max iterations reached

        Args:
            audit_fn: Function that returns AuditReport
            training_method: Override training method
            max_iterations: Override max iterations
            violation_threshold: Override violation threshold

        Returns:
            List of IterationResults for all iterations
        """
        max_iter = max_iterations or self.config.max_iterations
        threshold = violation_threshold or self.config.violation_threshold
        method = training_method or self.config.training_method

        logger.info(
            f"Starting convergence loop: max_iter={max_iter}, "
            f"threshold={threshold}, method={method}"
        )

        results = []
        prev_violation_rate = float('inf')

        for i in range(max_iter):
            # Run audit
            report = audit_fn()

            # Run training iteration
            result = self.run_iteration(
                audit_report=report,
                training_method=method,
                re_audit_fn=audit_fn,
            )
            results.append(result)

            # Check convergence
            if result.violations_after:
                current_rate = sum(result.violations_after.values())

                # Check if below threshold
                if current_rate < threshold:
                    logger.info(f"Converged: violation rate {current_rate:.4f} < {threshold}")
                    break

                # Check for improvement
                improvement = prev_violation_rate - current_rate
                if improvement < self.config.improvement_threshold:
                    logger.info(
                        f"Early stopping: improvement {improvement:.4f} < "
                        f"{self.config.improvement_threshold}"
                    )
                    break

                prev_violation_rate = current_rate

            # Check if no training happened
            if not result.training_completed:
                logger.info("No training performed, stopping loop")
                break

        logger.info(f"Convergence loop completed after {len(results)} iterations")
        return results

    def _compute_violation_rates(
        self,
        audit_report: Any,
    ) -> Dict[str, float]:
        """Extract violation rates from audit report."""
        rates = {
            "sufficiency": 0.0,
            "idempotence": 0.0,
            "merge": 0.0,
        }

        # Adapt based on actual AuditReport structure
        if hasattr(audit_report, 'violation_rates'):
            rates.update(audit_report.violation_rates)
        elif hasattr(audit_report, 'stats'):
            stats = audit_report.stats
            if hasattr(stats, 'sufficiency_violation_rate'):
                rates["sufficiency"] = stats.sufficiency_violation_rate
            if hasattr(stats, 'idempotence_violation_rate'):
                rates["idempotence"] = stats.idempotence_violation_rate
            if hasattr(stats, 'merge_violation_rate'):
                rates["merge"] = stats.merge_violation_rate

        return rates

    @property
    def history(self) -> List[IterationResult]:
        """Get all iteration results."""
        return self._iteration_history.copy()


# =============================================================================
# Convenience Functions
# =============================================================================

def run_audit_driven_training(
    audit_report: Any,
    model_name: str,
    judge: Any,
    summarizer: Any,
    output_dir: Union[str, Path],
    training_method: Literal["dpo", "grpo", "reward", "scalar_reward"] = "dpo",
) -> IterationResult:
    """
    Convenience function for single training iteration.

    Args:
        audit_report: AuditReport from auditor
        model_name: HuggingFace model name
        judge: GenRMJudge or similar
        summarizer: Summarizer module
        output_dir: Output directory
        training_method: Which trainer to use

    Returns:
        IterationResult
    """
    trainer = AuditDrivenTrainer(
        model_name=model_name,
        judge=judge,
        summarizer=summarizer,
        output_dir=output_dir,
    )
    return trainer.run_iteration(audit_report, training_method)
