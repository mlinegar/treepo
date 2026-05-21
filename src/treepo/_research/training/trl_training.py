"""
TRL Integration for Supervision-Based Training.

This module provides wrappers around TRL/HF training
trainers for DPO, GRPO, pairwise reward, and scalar reward model training. It bridges our preference
collection system with TRL's training infrastructure.

Dependencies:
    pip install trl>=0.7.0 transformers>=4.40.0 peft>=0.8.0

Architecture:
    SupervisionDataset → Optimizer Projection → HuggingFace Dataset → TRL Trainer

Usage:
    from treepo._research.training.trl_training import (
        train_dpo,
        train_grpo,
        train_reward_model,
        train_scalar_reward_model,
        TRLTrainingConfig,
    )

    # Load supervision data
    dataset = SupervisionDataset.load("supervision.json")

    # Train DPO
    train_dpo(
        dataset=dataset,
        model_name="nvidia/Nemotron-Nano-8B",
        output_dir="models/dpo_trained",
        config=TRLTrainingConfig(
            learning_rate=1e-5,
            num_train_epochs=3,
            use_lora=True,
        ),
    )

    # Train pairwise reward model
    train_reward_model(
        dataset=dataset,
        model_name="nvidia/Nemotron-Nano-8B",
        output_dir="models/reward_model",
    )

    # Train scalar reward regressor
    train_scalar_reward_model(
        dataset=dataset,
        model_name="nvidia/Nemotron-Nano-8B",
        output_dir="models/scalar_reward_model",
    )
"""

import logging
import inspect
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union

from treepo._research.training.supervision.adapters import (
    build_dpo_training_records,
    build_group_grpo_training_records,
    build_reward_model_training_records,
    build_scalar_reward_training_records,
)
from treepo._research.training.supervision import (
    BinaryComparison,
    BinaryProjectionDataset,
    ComparativeDataset,
    ComparativeJudgment,
    PreferenceDataset,
    PromptBuilder,
    SupervisionDataset,
)
from treepo._research.training.supervision.optimizer_metadata import (
    TreePOWeightingMode,
    validate_discount_gamma,
    validate_tree_objective_weighting_mode,
)
from treepo._research.training.config_sections import (
    OptimizerConfig,
    RuntimeConfig,
    TrainConfig,
    ValidationConfig,
)
from treepo._research.stats.sampling import (
    largest_remainder_allocation as _largest_remainder_allocation,
    pps_inclusion_probabilities as _pps_inclusion_probabilities,
    systematic_pps_sample_indices as _systematic_pps_sample_indices,
)

logger = logging.getLogger(__name__)

TrainingSupervision = Union[
    SupervisionDataset,
    BinaryProjectionDataset,
    ComparativeDataset,
    Sequence[BinaryComparison],
    Sequence[ComparativeJudgment],
    PreferenceDataset,
    Sequence[BinaryComparison],
    Sequence[ComparativeJudgment],
]


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True, kw_only=True)
class TRLSequenceConfig:
    max_length: int = 2048
    max_prompt_length: int = 1024


@dataclass(frozen=True, kw_only=True)
class TRLLoraConfig:
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass(frozen=True, kw_only=True)
class TRLQuantizationConfig:
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"


@dataclass(frozen=True, kw_only=True)
class TRLDPOConfig:
    beta: float = 0.1


@dataclass(frozen=True, kw_only=True)
class TRLGRPOConfig:
    num_generations: int = 4


@dataclass(frozen=True, kw_only=True)
class TRLRewardObjectiveConfig:
    reward_use_margin: bool = False
    reward_margin_source: Literal["score_estimate", "oracle_error"] = "score_estimate"
    reward_margin_scale: Optional[float] = None
    scalar_reward_loss: Literal["mse", "smooth_l1", "l1"] = "mse"
    scalar_reward_huber_delta: float = 1.0


@dataclass(frozen=True, kw_only=True)
class TRLPropensityWeightingConfig:
    use_propensity_weighting: bool = True
    propensity_resample: bool = True
    propensity_native_loss_weighting: bool = True
    propensity_weight_clip: Optional[float] = None
    propensity_random_seed: int = 42
    propensity_sampling_strategy: Literal[
        "multinomial",
        "pps_systematic",
        "stratified_multinomial",
    ] = "pps_systematic"
    propensity_stratify_key: Optional[str] = "law_type"
    tree_objective_weighting_mode: TreePOWeightingMode = "legacy_channel"
    discount_gamma: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tree_objective_weighting_mode",
            validate_tree_objective_weighting_mode(self.tree_objective_weighting_mode),
        )
        object.__setattr__(
            self,
            "discount_gamma",
            validate_discount_gamma(self.discount_gamma),
        )


@dataclass(frozen=True, kw_only=True)
class TRLTrainingConfig:
    """Sectioned configuration for TRL-based training."""

    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            epochs=3,
            batch_size=2,
            gradient_accumulation_steps=8,
            logging_steps=10,
            save_steps=100,
        )
    )
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            learning_rate=1e-5,
            warmup_ratio=0.1,
        )
    )
    validation: ValidationConfig = field(
        default_factory=lambda: ValidationConfig(eval_steps=100)
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lora: TRLLoraConfig = field(default_factory=TRLLoraConfig)
    quantization: TRLQuantizationConfig = field(default_factory=TRLQuantizationConfig)
    sequence: TRLSequenceConfig = field(default_factory=TRLSequenceConfig)
    dpo: TRLDPOConfig = field(default_factory=TRLDPOConfig)
    grpo: TRLGRPOConfig = field(default_factory=TRLGRPOConfig)
    reward_objective: TRLRewardObjectiveConfig = field(
        default_factory=TRLRewardObjectiveConfig
    )
    propensity_weighting: TRLPropensityWeightingConfig = field(
        default_factory=TRLPropensityWeightingConfig
    )


# =============================================================================
# Dataset Conversion
# =============================================================================

def _extract_sample_weight(
    record: Dict[str, Any],
    default_weight: float = 1.0,
) -> float:
    """Extract sample weight from exported preference record."""
    if "sample_weight" in record:
        try:
            value = float(record["sample_weight"])
            if value >= 0:
                return value
        except (TypeError, ValueError):
            pass

    metadata = record.get("metadata") or {}
    if isinstance(metadata, dict) and "sample_weight" in metadata:
        try:
            value = float(metadata.get("sample_weight"))
            if value >= 0:
                return value
        except (TypeError, ValueError):
            pass

    treepo = metadata.get("treepo") if isinstance(metadata, dict) else None
    if isinstance(treepo, dict):
        for key in ("effective_weight", "sample_weight", "ipw_weight"):
            try:
                value = float(treepo.get(key))
                if value >= 0:
                    return value
            except (TypeError, ValueError):
                continue
        try:
            propensity = float(treepo.get("joint_propensity", 1.0))
            if propensity > 0:
                return 1.0 / propensity
        except (TypeError, ValueError):
            pass

    return default_weight


def _treepo_metadata_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record, dict) else None
    if isinstance(metadata, dict):
        treepo = metadata.get("treepo")
        if isinstance(treepo, dict):
            return treepo
    treepo = record.get("treepo") if isinstance(record, dict) else None
    if isinstance(treepo, dict):
        return treepo
    return {}


def _summarize_weight_values(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0.0, "min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "mean": float(sum(values) / len(values)),
        "max": float(max(values)),
    }


def _summarize_treepo_weighting(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    objective_weights: List[float] = []
    ipw_weights: List[float] = []
    effective_weights: List[float] = []
    depths: List[int] = []
    channels: Dict[str, int] = {}
    roles: Dict[str, int] = {}
    weighting_modes: Dict[str, int] = {}
    sample_weight_sources: Dict[str, int] = {}

    for record in records:
        treepo = _treepo_metadata_from_record(record)
        if not treepo:
            continue

        try:
            objective_weights.append(float(treepo.get("objective_weight", 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            ipw_weights.append(float(treepo.get("ipw_weight", 0.0)))
        except (TypeError, ValueError):
            pass
        try:
            effective_weights.append(float(treepo.get("effective_weight", _extract_sample_weight(record))))
        except (TypeError, ValueError):
            effective_weights.append(_extract_sample_weight(record))
        try:
            depths.append(int(treepo.get("depth", 0)))
        except (TypeError, ValueError):
            pass

        channel = str(treepo.get("channel", "unknown") or "unknown")
        channels[channel] = channels.get(channel, 0) + 1

        rl_role = treepo.get("rl_role")
        if rl_role is not None:
            role_key = str(rl_role)
            roles[role_key] = roles.get(role_key, 0) + 1

        weighting_mode = str(treepo.get("weighting_mode", "unknown") or "unknown")
        weighting_modes[weighting_mode] = weighting_modes.get(weighting_mode, 0) + 1

        sample_weight_source = str(
            treepo.get("sample_weight_source", "unknown") or "unknown"
        )
        sample_weight_sources[sample_weight_source] = (
            sample_weight_sources.get(sample_weight_source, 0) + 1
        )

    return {
        "objective_weight": _summarize_weight_values(objective_weights),
        "ipw_weight": _summarize_weight_values(ipw_weights),
        "effective_weight": _summarize_weight_values(effective_weights),
        "depth": _summarize_weight_values([float(depth) for depth in depths]),
        "channels": channels,
        "rl_roles": roles,
        "weighting_modes": weighting_modes,
        "sample_weight_sources": sample_weight_sources,
    }


def _log_treepo_weighting_summary(
    records: Sequence[Dict[str, Any]],
    *,
    trainer_name: str,
    config: TRLTrainingConfig,
) -> None:
    summary = _summarize_treepo_weighting(records)
    logger.info(
        "%s TreePO RL weighting: mode=%s gamma=%.4f sample_weight_source=effective_weight "
        "objective[min=%.4g mean=%.4g max=%.4g] "
        "ipw[min=%.4g mean=%.4g max=%.4g] "
        "effective[min=%.4g mean=%.4g max=%.4g] "
        "channels=%s roles=%s",
        trainer_name,
        config.propensity_weighting.tree_objective_weighting_mode,
        float(config.propensity_weighting.discount_gamma),
        summary["objective_weight"]["min"],
        summary["objective_weight"]["mean"],
        summary["objective_weight"]["max"],
        summary["ipw_weight"]["min"],
        summary["ipw_weight"]["mean"],
        summary["ipw_weight"]["max"],
        summary["effective_weight"]["min"],
        summary["effective_weight"]["mean"],
        summary["effective_weight"]["max"],
        summary["channels"],
        summary["rl_roles"],
    )


def _resample_records_by_weight(
    records: List[Dict[str, Any]],
    config: TRLTrainingConfig,
) -> List[Dict[str, Any]]:
    """
    Resample records by sample_weight when weighting is enabled.

    This provides weighting support for trainers that do not consume
    per-example weights natively.
    """
    if not config.propensity_weighting.use_propensity_weighting or not config.propensity_weighting.propensity_resample or not records:
        return records

    weights = [
        min(_extract_sample_weight(record), config.propensity_weighting.propensity_weight_clip)
        if config.propensity_weighting.propensity_weight_clip is not None
        else _extract_sample_weight(record)
        for record in records
    ]
    total_weight = sum(weights)
    if total_weight <= 0:
        return records

    strategy = config.propensity_weighting.propensity_sampling_strategy
    rng = random.Random(config.propensity_weighting.propensity_random_seed)
    size = len(records)

    if strategy == "multinomial":
        return rng.choices(records, weights=weights, k=size)

    if strategy == "pps_systematic":
        sum_w = sum(weights)
        sum_w_sq = sum(weight * weight for weight in weights)
        neff = int(round((sum_w * sum_w / sum_w_sq))) if sum_w_sq > 0 else 0
        base_size = max(1, min(len(records), neff))

        inclusion_probs = _pps_inclusion_probabilities(weights, base_size)
        sampled_indices = _systematic_pps_sample_indices(inclusion_probs, base_size, rng)
        sampled = [records[index] for index in sampled_indices]

        if len(sampled) < size:
            sampled.extend(rng.choices(records, weights=weights, k=size - len(sampled)))
        return sampled

    if strategy == "stratified_multinomial":
        stratify_key = config.propensity_weighting.propensity_stratify_key
        if not stratify_key:
            return rng.choices(records, weights=weights, k=size)

        groups: Dict[str, List[int]] = {}
        for index, record in enumerate(records):
            value = record.get(stratify_key)
            if value is None and isinstance(record.get("metadata"), dict):
                value = record["metadata"].get(stratify_key)
            key = str(value)
            groups.setdefault(key, []).append(index)

        if not groups:
            return rng.choices(records, weights=weights, k=size)

        keys = list(groups.keys())
        group_mass = [sum(weights[index] for index in groups[key]) for key in keys]
        total_mass = sum(group_mass)
        if total_mass <= 0:
            return rng.choices(records, k=size)

        quotas = [size * (mass / total_mass) for mass in group_mass]
        allocation = _largest_remainder_allocation(size, quotas)

        sampled: List[Dict[str, Any]] = []
        for key, alloc in zip(keys, allocation):
            if alloc <= 0:
                continue
            group_indices = groups[key]
            group_records = [records[index] for index in group_indices]
            group_weights = [weights[index] for index in group_indices]
            sampled.extend(rng.choices(group_records, weights=group_weights, k=alloc))
        return sampled

    logger.warning(
        "Unknown propensity sampling strategy '%s'; falling back to multinomial",
        strategy,
    )
    return rng.choices(records, weights=weights, k=size)


def _build_processing_class_kwargs(
    trainer_cls: Any,
    processing_class: Any,
) -> Dict[str, Any]:
    """
    Return trainer kwargs for tokenizer/processing_class across TRL versions.

    Newer TRL trainers use `processing_class=...`; older releases used
    `tokenizer=...`.
    """
    init_params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in init_params:
        return {"processing_class": processing_class}
    if "tokenizer" in init_params:
        return {"tokenizer": processing_class}
    return {}

def _preference_to_hf_dpo(
    preference_data: List[Dict[str, Any]],
) -> "Dataset":
    """
    Convert DPO format data to HuggingFace Dataset.

    Args:
        preference_data: Output from PreferenceDataset.to_preference_format("dpo")

    Returns:
        HuggingFace Dataset with prompt, chosen, rejected columns
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    # Filter out ties (no chosen/rejected for ties)
    filtered = [
        {
            "prompt": d["prompt"],
            "chosen": d["chosen"],
            "rejected": d["rejected"],
            "sample_weight": _extract_sample_weight(d),
            "metadata": d.get("metadata", {}),
            "preference_supervision": dict(
                dict(d.get("metadata", {}) or {}).get("preference_supervision", {}) or {}
            ),
            "comparative_signal": dict(
                dict(d.get("metadata", {}) or {}).get("comparative_signal", {}) or {}
            ),
        }
        for d in preference_data
        if d.get("chosen") and d.get("rejected")
    ]

    logger.info(f"Converted {len(filtered)} preference pairs to DPO format")
    return Dataset.from_list(filtered)


def _concat_prompt_response(prompt: str, response: str) -> str:
    """Join prompt and response into a single sequence."""
    if not prompt:
        return response
    if prompt.endswith(("\n", " ")):
        return f"{prompt}{response}"
    return f"{prompt}\n{response}"


def _preference_to_hf_reward(
    reward_pairs: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> "Dataset":
    """
    Convert chosen/rejected pairs into RewardTrainer tokenized format.

    Args:
        reward_pairs: List with prompt/chosen/rejected and optional margin
        tokenizer: HuggingFace tokenizer
        max_length: Max sequence length for tokenization

    Returns:
        HuggingFace Dataset with input_ids_* and attention_mask_* fields
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    converted = []
    for pair in reward_pairs:
        chosen_text = _concat_prompt_response(pair["prompt"], pair["chosen"])
        rejected_text = _concat_prompt_response(pair["prompt"], pair["rejected"])

        chosen_enc = tokenizer(chosen_text, truncation=True, max_length=max_length)
        rejected_enc = tokenizer(rejected_text, truncation=True, max_length=max_length)

        entry = {
            "input_ids_chosen": chosen_enc["input_ids"],
            "attention_mask_chosen": chosen_enc["attention_mask"],
            "input_ids_rejected": rejected_enc["input_ids"],
            "attention_mask_rejected": rejected_enc["attention_mask"],
            "sample_weight": float(pair.get("sample_weight", 1.0)),
            "metadata": dict(pair.get("metadata", {}) or {}),
            "preference_supervision": dict(pair.get("preference_supervision", {}) or {}),
            "comparative_signal": dict(pair.get("comparative_signal", {}) or {}),
        }
        if pair.get("margin") is not None:
            entry["margin"] = pair["margin"]
        converted.append(entry)

    logger.info(f"Converted {len(converted)} preference pairs to reward model format")
    return Dataset.from_list(converted)


def _scalar_reward_to_hf_dataset(
    reward_records: List[Dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> "Dataset":
    """Convert scalar reward rows into a tokenized HuggingFace Dataset."""
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    converted: List[Dict[str, Any]] = []
    for record in reward_records:
        prompt = str(record.get("prompt", "") or "")
        response = str(record.get("response", "") or "")
        score = record.get("score")
        if not response or score is None:
            continue
        encoded = tokenizer(
            _concat_prompt_response(prompt, response),
            truncation=True,
            max_length=max_length,
        )
        converted.append(
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": float(score),
                "sample_weight": _extract_sample_weight(record),
            }
        )

    logger.info("Converted %d scalar reward records to tokenized format", len(converted))
    return Dataset.from_list(converted)


def _preference_to_hf_grpo(
    grpo_data: List[Dict[str, Any]],
) -> "Dataset":
    """
    Convert GRPO format to HuggingFace Dataset.

    Args:
        grpo_data: Output from PreferenceDataset.to_grouped_grpo_format()

    Returns:
        HuggingFace Dataset with prompt, responses, ranks columns
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    converted = [
        {
            "prompt": d["prompt"],
            "responses": d["responses"],
            "ranks": d["ranks"],
            "scores": d.get("scores"),
            "sample_weight": _extract_sample_weight(d),
            "metadata": dict(d.get("metadata", {}) or {}),
            "preference_supervision": dict(
                dict(d.get("metadata", {}) or {}).get("preference_supervision", {}) or {}
            ),
            "comparative_signal": dict(
                dict(d.get("metadata", {}) or {}).get("comparative_signal", {}) or {}
            ),
        }
        for d in grpo_data
    ]

    logger.info(f"Converted {len(converted)} groups to GRPO format")
    return Dataset.from_list(converted)


def _compute_reward_margin(
    chosen_score: Optional[float],
    rejected_score: Optional[float],
    chosen_error: Optional[float],
    rejected_error: Optional[float],
    config: TRLTrainingConfig,
) -> Optional[float]:
    """Compute optional margin for reward modeling."""
    if not config.reward_objective.reward_use_margin:
        return None

    margin = None
    if config.reward_objective.reward_margin_source == "oracle_error":
        if chosen_error is not None and rejected_error is not None:
            margin = rejected_error - chosen_error
    else:
        if chosen_score is not None and rejected_score is not None:
            margin = chosen_score - rejected_score

    if margin is None:
        return None

    if config.reward_objective.reward_margin_scale:
        margin = margin / config.reward_objective.reward_margin_scale

    if margin <= 0:
        return None

    return margin


# =============================================================================
# Model Loading Utilities
# =============================================================================

def _load_model_for_training(
    model_name: str,
    config: TRLTrainingConfig,
    is_reward_model: bool = False,
):
    """
    Load model with optional quantization and LoRA.

    Args:
        model_name: HuggingFace model name or path
        config: Training configuration
        is_reward_model: Whether loading for reward model training

    Returns:
        Tuple of (model, tokenizer, peft_config or None)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    # Determine compute dtype
    compute_dtype = getattr(torch, config.quantization.bnb_4bit_compute_dtype)

    # Quantization config
    quantization_config = None
    if config.quantization.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, skipping quantization")
            quantization_config = None

    # Load model
    model_cls = AutoModelForSequenceClassification if is_reward_model else AutoModelForCausalLM
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }
    if is_reward_model:
        model_kwargs["num_labels"] = 1
        model_kwargs["ignore_mismatched_sizes"] = True

    model = model_cls.from_pretrained(model_name, **model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    peft_config = None
    if config.lora.use_lora:
        try:
            from peft import LoraConfig, TaskType
            peft_config = LoraConfig(
                r=config.lora.lora_r,
                lora_alpha=config.lora.lora_alpha,
                lora_dropout=config.lora.lora_dropout,
                target_modules=config.lora.lora_target_modules,
                task_type=TaskType.SEQ_CLS if is_reward_model else TaskType.CAUSAL_LM,
            )
        except ImportError:
            logger.warning("peft not available, training without LoRA")

    return model, tokenizer, peft_config


# =============================================================================
# Training Functions
# =============================================================================

def _coerce_sample_weight_tensor(
    raw_weights: Any,
    batch_size: int,
    device: Any,
):
    """Convert batch sample weights to a nonnegative tensor or return None."""
    if raw_weights is None:
        return None

    import torch

    if torch.is_tensor(raw_weights):
        weights = raw_weights.to(device=device, dtype=torch.float32)
    else:
        try:
            weights = torch.tensor(raw_weights, dtype=torch.float32, device=device)
        except Exception:
            return None

    weights = weights.reshape(-1)
    if weights.numel() == 1 and batch_size > 1:
        weights = weights.expand(batch_size)
    if weights.numel() != batch_size:
        return None
    weights = torch.clamp(weights, min=0.0)
    if float(weights.sum().item()) <= 0:
        return None
    return weights


def _build_weighted_dpo_trainer(base_cls):
    """Create a DPOTrainer subclass that applies per-example sample weights."""
    import torch

    class WeightedDPOTrainer(base_cls):
        def _weighted_reduce(self, values: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
            values = self._per_example_mean(values)
            if weights is None:
                return values.mean()
            denom = weights.sum().clamp(min=1e-12)
            return (values * weights).sum() / denom

        def _per_example_mean(self, values: torch.Tensor) -> torch.Tensor:
            if values.ndim == 0:
                return values.reshape(1)
            if values.ndim <= 1:
                return values
            return values.reshape(values.shape[0], -1).mean(dim=1)

        def get_batch_loss_metrics(self, model, batch, train_eval: str = "train"):
            metrics = {}
            prefix = "eval_" if train_eval == "eval" else ""

            model_output = self.concatenated_forward(model, batch)
            if isinstance(model_output, dict):
                if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                    reference_chosen_logps = batch["ref_chosen_logps"]
                    reference_rejected_logps = batch["ref_rejected_logps"]
                elif "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                    reference_chosen_logps = batch["reference_chosen_logps"]
                    reference_rejected_logps = batch["reference_rejected_logps"]
                else:
                    reference_chosen_logps, reference_rejected_logps = self.compute_ref_log_probs(batch)

                losses = 0
                chosen_rewards = 0
                rejected_rewards = 0
                loss_types = self.loss_type if isinstance(self.loss_type, (list, tuple)) else [self.loss_type]
                loss_weights = getattr(self, "loss_weights", None)
                for index, loss_type in enumerate(loss_types):
                    _losses, _chosen_rewards, _rejected_rewards = self.dpo_loss(
                        model_output["chosen_logps"],
                        model_output["rejected_logps"],
                        reference_chosen_logps,
                        reference_rejected_logps,
                        loss_type,
                        model_output,
                    )
                    weight = loss_weights[index] if loss_weights else 1.0
                    losses = losses + _losses * weight
                    chosen_rewards = chosen_rewards + _chosen_rewards * weight
                    rejected_rewards = rejected_rewards + _rejected_rewards * weight

                if getattr(self.args, "rpo_alpha", None) is not None and "nll_loss" in model_output:
                    losses = losses + self.args.rpo_alpha * model_output["nll_loss"]

                if getattr(self, "use_weighting", False) and "policy_weights" in model_output:
                    losses = losses * model_output["policy_weights"]

                if getattr(self, "aux_loss_enabled", False) and "aux_loss" in model_output:
                    losses = losses + self.aux_loss_coef * model_output["aux_loss"]

                batch_size = model_output["chosen_logps"].shape[0]
                weights = _coerce_sample_weight_tensor(
                    batch.get("sample_weight"),
                    batch_size=batch_size,
                    device=model_output["chosen_logps"].device,
                )
                loss = self._weighted_reduce(losses, weights)

                reward_accuracies = (chosen_rewards > rejected_rewards).float()

                metrics[f"{prefix}rewards/chosen"] = float(
                    self._weighted_reduce(chosen_rewards.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/rejected"] = float(
                    self._weighted_reduce(rejected_rewards.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/accuracies"] = float(
                    self._weighted_reduce(reward_accuracies.detach(), weights).cpu().item()
                )
                metrics[f"{prefix}rewards/margins"] = float(
                    self._weighted_reduce((chosen_rewards - rejected_rewards).detach(), weights).cpu().item()
                )
                metrics[f"{prefix}logps/chosen"] = float(
                    self._weighted_reduce(model_output["chosen_logps"].detach(), weights).cpu().item()
                )
                metrics[f"{prefix}logps/rejected"] = float(
                    self._weighted_reduce(model_output["rejected_logps"].detach(), weights).cpu().item()
                )
                if "mean_chosen_logits" in model_output:
                    metrics[f"{prefix}logits/chosen"] = float(
                        self._weighted_reduce(model_output["mean_chosen_logits"].detach(), weights).cpu().item()
                    )
                if "mean_rejected_logits" in model_output:
                    metrics[f"{prefix}logits/rejected"] = float(
                        self._weighted_reduce(model_output["mean_rejected_logits"].detach(), weights).cpu().item()
                    )
                if getattr(self.args, "rpo_alpha", None) is not None and "nll_loss" in model_output:
                    metrics[f"{prefix}nll_loss"] = float(
                        self._weighted_reduce(model_output["nll_loss"].detach(), weights).cpu().item()
                    )
                if getattr(self, "aux_loss_enabled", False) and "aux_loss" in model_output:
                    metrics[f"{prefix}aux_loss"] = float(
                        self._weighted_reduce(model_output["aux_loss"].detach(), weights).cpu().item()
                    )

                return loss, metrics

            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
            ) = model_output

            if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
                reference_chosen_logps = batch["reference_chosen_logps"]
                reference_rejected_logps = batch["reference_rejected_logps"]
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.null_ref_context():
                            (
                                reference_chosen_logps,
                                reference_rejected_logps,
                                _,
                                _,
                            ) = self.concatenated_forward(self.model, batch)
                    else:
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.ref_model, batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )
            weights = _coerce_sample_weight_tensor(
                batch.get("sample_weight"),
                batch_size=losses.shape[0],
                device=losses.device,
            )
            loss = self._weighted_reduce(losses, weights)

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            metrics[f"{prefix}rewards/chosen"] = float(self._weighted_reduce(
                chosen_rewards.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/rejected"] = float(self._weighted_reduce(
                rejected_rewards.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/accuracies"] = float(self._weighted_reduce(
                reward_accuracies.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}rewards/margins"] = float(self._weighted_reduce(
                (chosen_rewards - rejected_rewards).detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logps/rejected"] = float(self._weighted_reduce(
                policy_rejected_logps.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logps/chosen"] = float(self._weighted_reduce(
                policy_chosen_logps.detach(),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logits/rejected"] = float(self._weighted_reduce(
                self._per_example_mean(policy_rejected_logits.detach()),
                weights,
            ).cpu().item())
            metrics[f"{prefix}logits/chosen"] = float(self._weighted_reduce(
                self._per_example_mean(policy_chosen_logits.detach()),
                weights,
            ).cpu().item())

            return loss, metrics

    return WeightedDPOTrainer


def _build_weighted_reward_data_collator(tokenizer: Any, max_length: Optional[int]):
    """Create a RewardTrainer data collator that preserves sample_weight."""
    import torch
    base_collator = None
    try:
        from trl.trainer.reward_trainer import DataCollatorForPreference

        base_collator = DataCollatorForPreference(
            pad_token_id=tokenizer.pad_token_id,
            return_tensors="pt",
        )
    except Exception:
        base_collator = None

    class WeightedRewardDataCollator:
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            sample_weights = torch.tensor(
                [float(feature.get("sample_weight", 1.0)) for feature in features],
                dtype=torch.float32,
            )

            # TRL >=0.26 reward format (chosen_input_ids/rejected_input_ids)
            if "chosen_input_ids" in features[0] and "rejected_input_ids" in features[0]:
                if base_collator is not None:
                    batch = base_collator(features)
                else:
                    chosen_input_ids = [torch.tensor(feature["chosen_input_ids"]) for feature in features]
                    rejected_input_ids = [torch.tensor(feature["rejected_input_ids"]) for feature in features]
                    input_ids = chosen_input_ids + rejected_input_ids
                    attention_mask = [torch.ones_like(ids) for ids in input_ids]
                    input_ids = tokenizer.pad(
                        {"input_ids": input_ids},
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )["input_ids"]
                    attention_mask = tokenizer.pad(
                        {"input_ids": attention_mask},
                        padding=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )["input_ids"]
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                    }
                    if "margin" in features[0]:
                        batch["margin"] = torch.tensor(
                            [float(feature["margin"]) for feature in features],
                            dtype=torch.float32,
                        )
                batch["sample_weight"] = sample_weights
                return batch

            # Legacy format (already tokenized chosen/rejected pairs)
            features_chosen = []
            features_rejected = []
            margins: List[float] = []

            has_margin = "margin" in features[0]
            for feature in features:
                features_chosen.append(
                    {
                        "input_ids": feature["input_ids_chosen"],
                        "attention_mask": feature["attention_mask_chosen"],
                    }
                )
                features_rejected.append(
                    {
                        "input_ids": feature["input_ids_rejected"],
                        "attention_mask": feature["attention_mask_rejected"],
                    }
                )
                if has_margin:
                    margins.append(float(feature["margin"]))

            batch_chosen = tokenizer.pad(
                features_chosen,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_rejected = tokenizer.pad(
                features_rejected,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )

            batch = {
                "input_ids_chosen": batch_chosen["input_ids"],
                "attention_mask_chosen": batch_chosen["attention_mask"],
                "input_ids_rejected": batch_rejected["input_ids"],
                "attention_mask_rejected": batch_rejected["attention_mask"],
                "return_loss": True,
                "sample_weight": sample_weights,
            }
            if has_margin:
                batch["margin"] = torch.tensor(margins, dtype=torch.float32)
            return batch

    return WeightedRewardDataCollator()


def _build_weighted_reward_trainer(base_cls):
    """Create a RewardTrainer subclass that applies per-example sample weights."""
    import torch.nn as nn

    class WeightedRewardTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if "input_ids" in inputs and "attention_mask" in inputs:
                model_inputs = {key: value for key, value in inputs.items() if key != "sample_weight"}
                model_inputs["use_cache"] = False
                outputs = model(**model_inputs)
                rewards_chosen, rewards_rejected = outputs.logits.squeeze(-1).chunk(2)
                margin = inputs.get("margin")
                if margin is not None:
                    margin = margin.to(device=rewards_chosen.device, dtype=rewards_chosen.dtype)
                    per_example_loss = -nn.functional.logsigmoid(
                        rewards_chosen - rewards_rejected - margin
                    )
                else:
                    per_example_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

                weights = _coerce_sample_weight_tensor(
                    inputs.get("sample_weight"),
                    batch_size=per_example_loss.shape[0],
                    device=per_example_loss.device,
                )
                if weights is None:
                    loss = per_example_loss.mean()
                else:
                    denom = weights.sum().clamp(min=1e-12)
                    loss = (per_example_loss * weights).sum() / denom

                if getattr(self.args, "center_rewards_coefficient", None) is not None:
                    loss = loss + self.args.center_rewards_coefficient * torch.mean(
                        (rewards_chosen + rewards_rejected) ** 2
                    )

                if return_outputs:
                    return loss, outputs
                return loss

            rewards_chosen = model(
                input_ids=inputs["input_ids_chosen"],
                attention_mask=inputs["attention_mask_chosen"],
                return_dict=True,
            )["logits"].squeeze(-1)
            rewards_rejected = model(
                input_ids=inputs["input_ids_rejected"],
                attention_mask=inputs["attention_mask_rejected"],
                return_dict=True,
            )["logits"].squeeze(-1)

            margin = inputs.get("margin")
            if margin is not None:
                margin = margin.to(device=rewards_chosen.device, dtype=rewards_chosen.dtype)
                per_example_loss = -nn.functional.logsigmoid(
                    rewards_chosen - rewards_rejected - margin
                )
            else:
                per_example_loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected)

            weights = _coerce_sample_weight_tensor(
                inputs.get("sample_weight"),
                batch_size=per_example_loss.shape[0],
                device=per_example_loss.device,
            )
            if weights is None:
                loss = per_example_loss.mean()
            else:
                denom = weights.sum().clamp(min=1e-12)
                loss = (per_example_loss * weights).sum() / denom

            if return_outputs:
                return loss, {
                    "rewards_chosen": rewards_chosen,
                    "rewards_rejected": rewards_rejected,
                }
            return loss

    return WeightedRewardTrainer


def _build_scalar_reward_data_collator(tokenizer: Any, max_length: Optional[int]):
    """Create a scalar-reward data collator that preserves sample_weight."""
    import torch

    class ScalarRewardDataCollator:
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch = tokenizer.pad(
                [
                    {
                        "input_ids": feature["input_ids"],
                        "attention_mask": feature["attention_mask"],
                    }
                    for feature in features
                ],
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch["labels"] = torch.tensor(
                [float(feature["labels"]) for feature in features],
                dtype=torch.float32,
            )
            batch["sample_weight"] = torch.tensor(
                [float(feature.get("sample_weight", 1.0)) for feature in features],
                dtype=torch.float32,
            )
            return batch

    return ScalarRewardDataCollator()


def _build_weighted_scalar_reward_trainer(
    base_cls,
    *,
    loss_name: Literal["mse", "smooth_l1", "l1"] = "mse",
    huber_delta: float = 1.0,
):
    """Create a Trainer subclass for scalar reward regression with IPW weights."""
    import torch
    import torch.nn.functional as F

    class WeightedScalarRewardTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            labels = inputs.pop("labels")
            raw_weights = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).reshape(-1)
            labels = labels.to(device=logits.device, dtype=logits.dtype).reshape(-1)

            if loss_name == "mse":
                per_example_loss = (logits - labels) ** 2
            elif loss_name == "l1":
                per_example_loss = torch.abs(logits - labels)
            elif loss_name == "smooth_l1":
                per_example_loss = F.smooth_l1_loss(
                    logits,
                    labels,
                    reduction="none",
                    beta=float(huber_delta),
                )
            else:
                raise ValueError(
                    f"Unknown scalar reward loss: {loss_name!r}. "
                    "Expected one of {'mse', 'smooth_l1', 'l1'}."
                )

            if bool(getattr(self, "apply_sample_weight", True)):
                weights = _coerce_sample_weight_tensor(
                    raw_weights,
                    batch_size=per_example_loss.shape[0],
                    device=per_example_loss.device,
                )
            else:
                weights = None
            if weights is None:
                loss = per_example_loss.mean()
            else:
                denom = weights.sum().clamp(min=1e-12)
                loss = (per_example_loss * weights).sum() / denom

            if return_outputs:
                return loss, outputs
            return loss

    return WeightedScalarRewardTrainer


def _build_weighted_grpo_trainer(base_cls):
    """Create a GRPOTrainer subclass that applies per-example sample weights."""
    import torch

    class WeightedGRPOTrainer(base_cls):
        @staticmethod
        def _coerce_local_sample_weights(
            raw_inputs: List[Dict[str, Any]],
            device: Any,
            dtype: Any,
        ) -> Optional[torch.Tensor]:
            values: List[float] = []
            for example in raw_inputs:
                try:
                    values.append(max(0.0, float(example.get("sample_weight", 1.0))))
                except (TypeError, ValueError, AttributeError):
                    values.append(1.0)

            if not values:
                return None

            weights = torch.tensor(values, device=device, dtype=dtype)
            if float(weights.sum().item()) <= 0:
                return None

            # Keep average scale near one to stabilize optimizer hyperparameters.
            return weights / weights.mean().clamp(min=1e-12)

        def _generate_and_score_completions(self, inputs):
            batch = super()._generate_and_score_completions(inputs)
            advantages = batch.get("advantages")
            if advantages is None:
                return batch

            sample_weights = self._coerce_local_sample_weights(
                raw_inputs=inputs,
                device=advantages.device,
                dtype=advantages.dtype,
            )
            if sample_weights is None:
                return batch

            if sample_weights.shape[0] != advantages.shape[0]:
                logger.warning(
                    "Skipping GRPO native sample weighting due to shape mismatch "
                    "(weights=%s, advantages=%s)",
                    tuple(sample_weights.shape),
                    tuple(advantages.shape),
                )
                return batch

            if advantages.ndim == 1:
                batch["advantages"] = advantages * sample_weights
            else:
                batch["advantages"] = advantages * sample_weights.unsqueeze(-1)
            batch["sample_weight"] = sample_weights
            return batch

    return WeightedGRPOTrainer


def train_dpo(
    dataset: TrainingSupervision,
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    ref_model_name: Optional[str] = None,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train model using Direct Preference Optimization (DPO).

    Args:
        dataset: Preference or comparative supervision records
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration (uses defaults if None)
        ref_model_name: Reference model (uses model_name if None)
        law_type: Optional filter for specific law type
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise ImportError("TRL library required. Install with: pip install trl>=0.7.0")

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting DPO training with model: {model_name}")

    # Convert preference data to HF format
    dpo_data = build_dpo_training_records(
        dataset,
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=config.propensity_weighting.tree_objective_weighting_mode,
        discount_gamma=config.propensity_weighting.discount_gamma,
    )
    _log_treepo_weighting_summary(dpo_data, trainer_name="DPO", config=config)
    if (
        config.propensity_weighting.use_propensity_weighting
        and config.propensity_weighting.propensity_resample
        and not config.propensity_weighting.propensity_native_loss_weighting
    ):
        dpo_data = _resample_records_by_weight(dpo_data, config)

    train_dataset = _preference_to_hf_dpo(dpo_data)

    # Load models
    model, tokenizer, peft_config = _load_model_for_training(model_name, config)

    # Reference model (for KL penalty)
    ref_model = None
    if ref_model_name and ref_model_name != model_name:
        ref_model, _, _ = _load_model_for_training(ref_model_name, config)

    # DPO config
    training_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.optimizer.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_ratio=config.optimizer.warmup_ratio,
        max_length=config.sequence.max_length,
        max_prompt_length=config.sequence.max_prompt_length,
        beta=config.dpo.beta,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
    )

    # Create trainer
    trainer_cls = DPOTrainer
    if config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_dpo_trainer(DPOTrainer)

    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        **_build_processing_class_kwargs(trainer_cls, tokenizer),
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"DPO training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def _build_grpo_train_records(
    dataset: TrainingSupervision,
    *,
    config: TRLTrainingConfig,
    law_type: Optional[str],
    prompt_builder: Optional[PromptBuilder],
) -> List[Dict[str, Any]]:
    """
    Build GRPO prompt records while preserving reward-context columns.

    Reward functions may rely on `reference_score`/`original_text`, so these
    fields must survive any de-duplication or resampling path.
    """
    prompt_records = build_group_grpo_training_records(
        dataset,
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=config.propensity_weighting.tree_objective_weighting_mode,
        discount_gamma=config.propensity_weighting.discount_gamma,
    )
    _log_treepo_weighting_summary(prompt_records, trainer_name="GRPO prompts", config=config)
    for record in prompt_records:
        sample_weight = float(record.get("sample_weight", 1.0))
        if config.propensity_weighting.propensity_weight_clip is not None:
            sample_weight = min(sample_weight, float(config.propensity_weighting.propensity_weight_clip))
        record["sample_weight"] = sample_weight

    if not prompt_records:
        return []

    if config.propensity_weighting.use_propensity_weighting:
        if config.propensity_weighting.propensity_resample and not config.propensity_weighting.propensity_native_loss_weighting:
            logger.info(
                "Using weighted prompt resampling fallback for GRPO (native weighting disabled)."
            )
            prompt_records = _resample_records_by_weight(prompt_records, config)
        elif config.propensity_weighting.propensity_native_loss_weighting:
            logger.info(
                "Using native GRPO sample-weighted advantages for propensity weighting."
            )
        return [
            {
                "prompt": str(record.get("prompt", "")),
                "responses": list(record.get("responses", []) or []),
                "ranks": list(record.get("ranks", []) or []),
                "scores": list(record.get("scores", []) or []),
                "k": record.get("k"),
                "sample_weight": float(record.get("sample_weight", 1.0)),
                "reference_score": record.get("reference_score"),
                "original_text": record.get("original_text"),
                "rubric": record.get("rubric"),
                "law_type": record.get("law_type"),
                "preference_supervision": dict(record.get("preference_supervision", {}) or {}),
                "comparative_signal": dict(record.get("comparative_signal", {}) or {}),
                "metadata": dict(record.get("metadata", {}) or {}),
            }
            for record in prompt_records
            if str(record.get("prompt", "")).strip()
        ]

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()
    for record in prompt_records:
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            continue
        reference_score = record.get("reference_score")
        original_text = str(record.get("original_text", "") or "")
        key = (prompt, reference_score, original_text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            {
                "prompt": prompt,
                "responses": list(record.get("responses", []) or []),
                "ranks": list(record.get("ranks", []) or []),
                "scores": list(record.get("scores", []) or []),
                "k": record.get("k"),
                "sample_weight": float(record.get("sample_weight", 1.0)),
                "reference_score": reference_score,
                "original_text": original_text,
                "rubric": record.get("rubric"),
                "law_type": record.get("law_type"),
                "preference_supervision": dict(record.get("preference_supervision", {}) or {}),
                "comparative_signal": dict(record.get("comparative_signal", {}) or {}),
                "metadata": dict(record.get("metadata", {}) or {}),
            }
        )
    return deduped


def train_grpo(
    dataset: TrainingSupervision,
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    law_type: Optional[str] = None,
    reward_funcs: Optional[Union[Callable, List[Callable]]] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train model using Group Relative Policy Optimization (GRPO).

    GRPO in TRL is an online method that generates completions and scores
    them using reward functions. It does not consume offline ranked groups.

    Args:
        dataset: Preference or comparative supervision used to extract prompts
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration
        law_type: Optional filter for specific law type
        reward_funcs: Reward function(s) compatible with TRL GRPOTrainer
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError:
        raise ImportError(
            "TRL library with GRPO support required. "
            "Install with: pip install trl>=0.8.0"
        )

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reward_funcs is None:
        raise ValueError(
            "GRPO training requires reward_funcs. TRL GRPOTrainer is online and "
            "does not consume offline ranked preference groups."
        )

    logger.info(f"Starting GRPO training with model: {model_name}")

    # Build prompt-only dataset from preferences
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    prompt_records = _build_grpo_train_records(
        dataset,
        config=config,
        law_type=law_type,
        prompt_builder=prompt_builder,
    )

    if not prompt_records:
        raise ValueError("No prompts available for GRPO training after filtering")
    train_dataset = Dataset.from_list(prompt_records)

    # Load model
    model, tokenizer, peft_config = _load_model_for_training(model_name, config)

    # GRPO config
    training_args = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.optimizer.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_ratio=config.optimizer.warmup_ratio,
        num_generations=config.grpo.num_generations,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
    )

    # Create trainer
    trainer_cls = GRPOTrainer
    if config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_grpo_trainer(GRPOTrainer)

    trainer = trainer_cls(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"GRPO training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def train_reward_model(
    dataset: TrainingSupervision,
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """
    Train a reward model from preference data.

    The reward model learns to assign higher reward to preferred responses.

    Args:
        dataset: Preference or comparative supervision records
        model_name: HuggingFace model name to fine-tune
        output_dir: Directory to save trained model
        config: Training configuration
        law_type: Optional filter for specific law type
        prompt_builder: Optional prompt builder for generating prompts

    Returns:
        Path to saved model
    """
    try:
        from trl import RewardConfig, RewardTrainer
    except ImportError:
        raise ImportError("TRL library required. Install with: pip install trl>=0.7.0")

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting reward model training with model: {model_name}")

    # Build reward pairs (chosen/rejected) from preference data
    reward_pairs = build_reward_model_training_records(
        dataset,
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=config.propensity_weighting.tree_objective_weighting_mode,
        discount_gamma=config.propensity_weighting.discount_gamma,
    )
    _log_treepo_weighting_summary(reward_pairs, trainer_name="Reward", config=config)
    for entry in reward_pairs:
        sample_weight = float(entry.get("sample_weight", 1.0))
        if config.propensity_weighting.propensity_weight_clip is not None:
            sample_weight = min(sample_weight, float(config.propensity_weighting.propensity_weight_clip))
        entry["sample_weight"] = sample_weight
        margin = _compute_reward_margin(
            entry.get("chosen_score"),
            entry.get("rejected_score"),
            entry.get("chosen_error"),
            entry.get("rejected_error"),
            config,
        )
        if margin is not None:
            entry["margin"] = margin

    if not reward_pairs:
        raise ValueError("No reward pairs available after filtering")

    if (
        config.propensity_weighting.use_propensity_weighting
        and config.propensity_weighting.propensity_resample
        and not config.propensity_weighting.propensity_native_loss_weighting
    ):
        reward_pairs = _resample_records_by_weight(reward_pairs, config)

    # Load model (as sequence classification model)
    model, tokenizer, peft_config = _load_model_for_training(
        model_name, config, is_reward_model=True
    )

    # RewardTrainer API compatibility: newer TRL expects raw chosen/rejected text
    # with `processing_class`, while older paths used pre-tokenized pair fields.
    trainer_cls = RewardTrainer
    if config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting:
        trainer_cls = _build_weighted_reward_trainer(RewardTrainer)
    processing_kwargs = _build_processing_class_kwargs(trainer_cls, tokenizer)
    uses_processing_class = "processing_class" in processing_kwargs

    if uses_processing_class:
        from datasets import Dataset

        train_dataset = Dataset.from_list(reward_pairs)
    else:
        train_dataset = _preference_to_hf_reward(
            reward_pairs,
            tokenizer=tokenizer,
            max_length=config.sequence.max_length,
        )

    # Reward training config
    training_args = RewardConfig(
        output_dir=str(output_dir),
        learning_rate=config.optimizer.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_ratio=config.optimizer.warmup_ratio,
        max_length=config.sequence.max_length,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
    )

    # Create trainer
    data_collator = None
    if config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting:
        data_collator = _build_weighted_reward_data_collator(
            tokenizer=tokenizer,
            max_length=config.sequence.max_length,
        )

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        **processing_kwargs,
        data_collator=data_collator,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting reward model training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"Reward model training complete. Model saved to {output_dir / 'final'}")

    return str(output_dir / "final")


def train_scalar_reward_model(
    dataset: TrainingSupervision,
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    law_type: Optional[str] = None,
    prompt_builder: Optional[PromptBuilder] = None,
) -> str:
    """Train a scalar reward regressor from response-level supervision."""
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting scalar reward model training with model: %s", model_name)

    scalar_records = build_scalar_reward_training_records(
        dataset,
        law_type=law_type,
        prompt_builder=prompt_builder,
        tree_objective_weighting_mode=config.propensity_weighting.tree_objective_weighting_mode,
        discount_gamma=config.propensity_weighting.discount_gamma,
    )
    _log_treepo_weighting_summary(
        scalar_records,
        trainer_name="Scalar reward",
        config=config,
    )
    for entry in scalar_records:
        sample_weight = float(entry.get("sample_weight", 1.0))
        if config.propensity_weighting.propensity_weight_clip is not None:
            sample_weight = min(sample_weight, float(config.propensity_weighting.propensity_weight_clip))
        entry["sample_weight"] = sample_weight

    if not scalar_records:
        raise ValueError("No scalar reward records available after filtering")

    apply_scalar_sample_weight = bool(
        config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting
    )
    if config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_resample and not apply_scalar_sample_weight:
        scalar_records = _resample_records_by_weight(scalar_records, config)

    model, tokenizer, peft_config = _load_model_for_training(
        model_name,
        config,
        is_reward_model=True,
    )
    if hasattr(model, "config"):
        model.config.problem_type = "regression"

    if peft_config is not None:
        try:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
        except ImportError:
            logger.warning("peft not available, training scalar reward model without LoRA")

    train_dataset = _scalar_reward_to_hf_dataset(
        scalar_records,
        tokenizer=tokenizer,
        max_length=config.sequence.max_length,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.optimizer.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_ratio=config.optimizer.warmup_ratio,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_cls = _build_weighted_scalar_reward_trainer(
        Trainer,
        loss_name=config.reward_objective.scalar_reward_loss,
        huber_delta=config.reward_objective.scalar_reward_huber_delta,
    )
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_build_scalar_reward_data_collator(
            tokenizer=tokenizer,
            max_length=config.sequence.max_length,
        ),
    )
    trainer.apply_sample_weight = apply_scalar_sample_weight

    logger.info("Starting scalar reward training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Scalar reward training complete. Model saved to %s", final_dir)
    return str(final_dir)


def train_scalar_reward_records(
    records: Sequence[Dict[str, Any]],
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    eval_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """Train a sequence-classification scalar regressor from exported rows.

    ``records`` are already in the scalar-reward wire format:
    ``{"prompt": str, "response": str, "score": float}``, with optional
    ``sample_weight``.  This is the lightweight bridge used by labeled-tree
    distillation when the scalar ``f`` student is a small LM rather than an
    embedding proxy.
    """
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError:
        raise ImportError(
            "transformers library required. Install with: pip install transformers"
        )

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scalar_records = [dict(record) for record in records]
    if not scalar_records:
        raise ValueError("No scalar reward records available after filtering")

    for entry in scalar_records:
        sample_weight = float(_extract_sample_weight(entry, default_weight=1.0))
        if config.propensity_weighting.propensity_weight_clip is not None:
            sample_weight = min(sample_weight, float(config.propensity_weighting.propensity_weight_clip))
        entry["sample_weight"] = sample_weight

    if (
        config.propensity_weighting.use_propensity_weighting
        and config.propensity_weighting.propensity_resample
        and not config.propensity_weighting.propensity_native_loss_weighting
    ):
        scalar_records = _resample_records_by_weight(scalar_records, config)

    apply_scalar_sample_weight = bool(
        config.propensity_weighting.use_propensity_weighting and config.propensity_weighting.propensity_native_loss_weighting
    )

    model, tokenizer, peft_config = _load_model_for_training(
        model_name,
        config,
        is_reward_model=True,
    )
    if hasattr(model, "config"):
        model.config.problem_type = "regression"

    if peft_config is not None:
        try:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
        except ImportError:
            logger.warning("peft not available, training scalar reward records without LoRA")

    train_dataset = _scalar_reward_to_hf_dataset(
        scalar_records,
        tokenizer=tokenizer,
        max_length=config.sequence.max_length,
    )
    eval_dataset = (
        _scalar_reward_to_hf_dataset(
            [dict(record) for record in eval_records],
            tokenizer=tokenizer,
            max_length=config.sequence.max_length,
        )
        if eval_records
        else None
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=config.optimizer.learning_rate,
        num_train_epochs=config.train.epochs,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        warmup_ratio=config.optimizer.warmup_ratio,
        logging_steps=config.train.logging_steps,
        save_steps=config.train.save_steps,
        eval_steps=config.validation.eval_steps if eval_dataset is not None else None,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
    )

    trainer_cls = _build_weighted_scalar_reward_trainer(
        Trainer,
        loss_name=config.reward_objective.scalar_reward_loss,
        huber_delta=config.reward_objective.scalar_reward_huber_delta,
    )
    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_build_scalar_reward_data_collator(
            tokenizer=tokenizer,
            max_length=config.sequence.max_length,
        ),
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = trainer_cls(**trainer_kwargs)
    trainer.apply_sample_weight = apply_scalar_sample_weight

    logger.info("Starting scalar reward record training...")
    trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Scalar reward record training complete. Model saved to %s", final_dir)
    return str(final_dir)


def train_sft(
    records: Sequence[Dict[str, str]],
    model_name: str,
    output_dir: Union[str, Path],
    config: Optional[TRLTrainingConfig] = None,
    eval_records: Optional[Sequence[Dict[str, str]]] = None,
) -> str:
    """Supervised fine-tuning via TRL's SFTTrainer with PEFT/LoRA + quantization.

    Parallel to `train_dpo` / `train_grpo` / `train_reward_model` —
    same `TRLTrainingConfig` controls use_lora, lora_r, target_modules,
    load_in_4bit, learning_rate, etc. Uses `_load_model_for_training` so
    SFT inherits identical LoRA and quantization handling as the preference
    path.

    `records` is a sequence of `{"prompt": str, "completion": str}` dicts
    (or `{"text": str}` directly). The train tensor is the concatenated
    "{prompt}\\n{completion}" text; TRL masks the prompt tokens in the loss.

    Returns the path to the saved final model.
    """
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        raise ImportError("TRL library required. Install with: pip install trl>=0.7.0")
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("datasets library required. Install with: pip install datasets")

    config = config or TRLTrainingConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting SFT training with model: {model_name}")

    def _to_text(rec: Dict[str, Any]) -> Dict[str, str]:
        if "text" in rec:
            return {"text": str(rec["text"])}
        prompt = str(rec.get("prompt", ""))
        completion = str(rec.get("completion", ""))
        return {"text": f"{prompt}\n{completion}"}

    train_ds = Dataset.from_list([_to_text(r) for r in records])
    eval_ds = (
        Dataset.from_list([_to_text(r) for r in eval_records])
        if eval_records else None
    )

    # Single source of truth: model + tokenizer + peft_config (NVFP4 / bitsandbytes
    # 4-bit + LoRA all handled here identically to the DPO/GRPO/reward paths).
    model, tokenizer, peft_config = _load_model_for_training(model_name, config)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        learning_rate=float(config.optimizer.learning_rate),
        num_train_epochs=int(config.train.epochs),
        per_device_train_batch_size=int(config.train.batch_size),
        gradient_accumulation_steps=int(config.train.gradient_accumulation_steps),
        warmup_ratio=float(config.optimizer.warmup_ratio),
        max_length=int(config.sequence.max_length),
        logging_steps=int(config.train.logging_steps),
        save_steps=int(config.train.save_steps),
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
    )

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        **_build_processing_class_kwargs(SFTTrainer, tokenizer),
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    if eval_ds is not None:
        trainer_kwargs["eval_dataset"] = eval_ds

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting SFT training...")
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    logger.info(f"SFT training complete. Model saved to {output_dir / 'final'}")
    return str(output_dir / "final")


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """CLI entry point for TRL training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train models using TRL/HF (DPO, GRPO, pairwise reward, scalar reward)"
    )
    parser.add_argument(
        "--method",
        choices=["dpo", "grpo", "reward", "scalar_reward"],
        required=True,
        help="Training method (grpo requires reward_funcs; see train_grpo)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to supervision JSON file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--law-type",
        type=str,
        default=None,
        help="Filter preferences by law type",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--tree-objective-weighting-mode",
        type=str,
        default="legacy_channel",
        choices=["legacy_channel", "discounted_tree"],
        help="How TreePO supervision exports map objective weights into TRL sample weights.",
    )
    parser.add_argument(
        "--discount-gamma",
        type=float,
        default=1.0,
        help="Depth-discount factor for discounted TreePO weighting mode.",
    )

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset from {args.dataset}")
    dataset = SupervisionDataset.load(args.dataset)

    # Create config
    config = TRLTrainingConfig(
        train=TrainConfig(epochs=args.epochs, batch_size=2, gradient_accumulation_steps=8),
        optimizer=OptimizerConfig(learning_rate=args.learning_rate, warmup_ratio=0.1),
        lora=TRLLoraConfig(use_lora=not args.no_lora),
        propensity_weighting=TRLPropensityWeightingConfig(
            tree_objective_weighting_mode=args.tree_objective_weighting_mode,
            discount_gamma=args.discount_gamma,
        ),
    )

    # Train
    if args.method == "dpo":
        train_dpo(dataset, args.model, args.output_dir, config, law_type=args.law_type)
    elif args.method == "grpo":
        train_grpo(dataset, args.model, args.output_dir, config, law_type=args.law_type)
    elif args.method == "reward":
        train_reward_model(dataset, args.model, args.output_dir, config, law_type=args.law_type)
    elif args.method == "scalar_reward":
        train_scalar_reward_model(
            dataset,
            args.model,
            args.output_dir,
            config,
            law_type=args.law_type,
        )


if __name__ == "__main__":
    main()
