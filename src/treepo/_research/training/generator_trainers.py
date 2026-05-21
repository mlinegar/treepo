"""
Pluggable Generator Trainers for Unified Training Loop.

This module provides a protocol and implementations for different generator
training methods. The unified trainer can use any of these methods interchangeably.

Supported Methods:
- DPO (Direct Preference Optimization): TRL-based, projects comparative supervision to pairs
- SFT (Supervised Fine-Tuning): TRL-based, uses tournament winners
- GRPO (Group Relative Policy Optimization): TRL-based, online with reward functions
- BootstrapFinetune: DSPy-based, teacher-student distillation

Usage:
    from treepo._research.training.generator_trainers import (
        get_trainer,
        DPOGeneratorTrainer,
        GRPOGeneratorTrainer,
        SFTGeneratorTrainer,
        BootstrapFinetuneTrainer,
    )

    # Get trainer by name
    trainer = get_trainer("dpo")
    model_path = trainer.train(preferences, model_name, output_dir)

    # Or instantiate directly with custom config
    trainer = GRPOGeneratorTrainer(genrm_judge=judge)
    model_path = trainer.train(preferences, model_name, output_dir, reward_funcs=reward_fn)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Tuple, Type, Union, runtime_checkable

from treepo._research.training.supervision import (
    BinaryProjectionDataset,
    SupervisionDataset,
    coerce_supervision_dataset,
    render_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Protocol Definition
# =============================================================================

@runtime_checkable
class GeneratorTrainer(Protocol):
    """
    Protocol for pluggable generator training methods.

    All generator trainers must implement the train() method that takes
    preference data and returns a path to the trained model.
    """

    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        **kwargs,
    ) -> str:
        """
        Train generator on preference data.

        Args:
            preferences: SupervisionDataset as the primary input surface
            model_name: HuggingFace model name to fine-tune
            output_dir: Directory to save trained model
            **kwargs: Method-specific arguments

        Returns:
            Path to saved model
        """
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeneratorTrainerConfig:
    """Base configuration for generator trainers."""

    # Training hyperparameters
    learning_rate: float = 1e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    max_length: int = 2048

    # LoRA configuration
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization
    load_in_4bit: bool = True
    bf16: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 100

    # Propensity/IPW weighting
    use_propensity_weighting: bool = True
    propensity_resample: bool = True
    propensity_native_loss_weighting: bool = True
    propensity_weight_clip: Optional[float] = None
    propensity_random_seed: int = 42
    propensity_sampling_strategy: str = "pps_systematic"
    propensity_stratify_key: Optional[str] = "law_type"


# =============================================================================
# Trainer Registry
# =============================================================================

_TRAINER_REGISTRY: Dict[str, Type["BaseGeneratorTrainer"]] = {}


def register_trainer(name: str):
    """Decorator to register a trainer class."""
    def decorator(cls: Type["BaseGeneratorTrainer"]):
        _TRAINER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_trainer(name: str, **kwargs) -> "BaseGeneratorTrainer":
    """
    Get a generator trainer by name.

    Args:
        name: Trainer name ("dpo", "sft", "grpo", "bootstrap_finetune")
        **kwargs: Arguments passed to trainer constructor

    Returns:
        Configured trainer instance

    Raises:
        ValueError: If trainer name is not registered
    """
    name_lower = name.lower()
    if name_lower not in _TRAINER_REGISTRY:
        available = list(_TRAINER_REGISTRY.keys())
        raise ValueError(f"Unknown trainer: '{name}'. Available: {available}")

    return _TRAINER_REGISTRY[name_lower](**kwargs)


def list_trainers() -> List[str]:
    """Return list of registered trainer names."""
    return list(_TRAINER_REGISTRY.keys())


# =============================================================================
# Base Trainer
# =============================================================================

class BaseGeneratorTrainer(ABC):
    """
    Abstract base class for generator trainers.

    Provides common functionality and defines the interface that all
    concrete trainers must implement.
    """

    def __init__(
        self,
        config: Optional[GeneratorTrainerConfig] = None,
        prompt_builder: Optional[Callable[[str, str], Any]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            prompt_builder: Optional prompt builder for generating prompts
        """
        self.config = config or GeneratorTrainerConfig()
        self.prompt_builder = prompt_builder

    @abstractmethod
    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        **kwargs,
    ) -> str:
        """
        Train generator on preference data.

        Must be implemented by concrete trainers.
        """
        ...

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the method name for logging."""
        ...

    def _prepare_pairs(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        law_type: Optional[str] = None,
    ) -> List[Any]:
        """
        Filter and optionally propensity-resample preference pairs.

        The global default propensity is uniform (all ones), so resampling is
        a no-op in the absence of logged non-uniform propensities.
        """
        projected_dataset = coerce_supervision_dataset(preferences).project_binary(
            projection="adjacent"
        )

        pairs = [
            pair for pair in projected_dataset.pairs
            if pair.preferred != "tie" and (law_type is None or pair.law_type == law_type)
        ]
        if not pairs:
            return []

        if not self.config.use_propensity_weighting or not self.config.propensity_resample:
            return pairs

        sampled = BinaryProjectionDataset(comparisons=pairs).sample_by_propensity(
            target_size=len(pairs),
            seed=self.config.propensity_random_seed,
            max_weight=self.config.propensity_weight_clip,
            strategy=self.config.propensity_sampling_strategy,
            stratify_by=self.config.propensity_stratify_key,
        )
        return sampled.comparisons


# =============================================================================
# DPO Trainer
# =============================================================================

@register_trainer("dpo")
class DPOGeneratorTrainer(BaseGeneratorTrainer):
    """
    TRL DPO - train on preference pairs.

    Uses Direct Preference Optimization which trains the model to prefer
    chosen responses over rejected responses using a sigmoid loss.
    """

    @property
    def method_name(self) -> str:
        return "DPO"

    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        law_type: Optional[str] = None,
        ref_model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Train using DPO.

        Args:
            preferences: SupervisionDataset or binary projection dataset
            model_name: HuggingFace model name to fine-tune
            output_dir: Directory to save trained model
            law_type: Optional filter for specific law type
            ref_model_name: Reference model for KL penalty (defaults to model_name)
            **kwargs: Additional arguments passed to TRL

        Returns:
            Path to saved model
        """
        from treepo._research.training.config_sections import OptimizerConfig, RuntimeConfig, TrainConfig
        from treepo._research.training.trl_training import (
            TRLLoraConfig,
            TRLPropensityWeightingConfig,
            TRLQuantizationConfig,
            TRLSequenceConfig,
            TRLTrainingConfig,
            train_dpo,
        )

        logger.info(f"Starting {self.method_name} training with model: {model_name}")

        # Create TRL config from our config
        trl_config = TRLTrainingConfig(
            train=TrainConfig(
                epochs=self.config.num_train_epochs,
                batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
            ),
            optimizer=OptimizerConfig(
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
            ),
            runtime=RuntimeConfig(bf16=self.config.bf16, gradient_checkpointing=True),
            sequence=TRLSequenceConfig(max_length=self.config.max_length),
            lora=TRLLoraConfig(
                use_lora=self.config.use_lora,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                lora_target_modules=self.config.lora_target_modules,
            ),
            quantization=TRLQuantizationConfig(load_in_4bit=self.config.load_in_4bit),
            propensity_weighting=TRLPropensityWeightingConfig(
                use_propensity_weighting=self.config.use_propensity_weighting,
                propensity_resample=self.config.propensity_resample,
                propensity_native_loss_weighting=self.config.propensity_native_loss_weighting,
                propensity_weight_clip=self.config.propensity_weight_clip,
                propensity_random_seed=self.config.propensity_random_seed,
                propensity_sampling_strategy=self.config.propensity_sampling_strategy,
                propensity_stratify_key=self.config.propensity_stratify_key,
            ),
        )

        return train_dpo(
            dataset=preferences,
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            ref_model_name=ref_model_name,
            law_type=law_type,
            prompt_builder=self.prompt_builder,
        )


# =============================================================================
# SFT Trainer
# =============================================================================

@register_trainer("sft")
class SFTGeneratorTrainer(BaseGeneratorTrainer):
    """
    TRL SFT - train on tournament winners.

    Uses supervised fine-tuning on the winning summaries from tournaments.
    This is simpler than DPO but doesn't use preference information.
    """

    @property
    def method_name(self) -> str:
        return "SFT"

    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        law_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Train using SFT on tournament winners.

        Args:
            preferences: SupervisionDataset or binary projection dataset
            model_name: HuggingFace model name to fine-tune
            output_dir: Directory to save trained model
            law_type: Optional filter for specific law type
            **kwargs: Additional arguments

        Returns:
            Path to saved model
        """
        try:
            from trl import SFTConfig, SFTTrainer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from datasets import Dataset
        except ImportError:
            raise ImportError(
                "TRL library required. Install with: pip install trl>=0.7.0"
            )

        logger.info(f"Starting {self.method_name} training with model: {model_name}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract winners from preferences
        sft_data = []
        for pair in self._prepare_pairs(preferences, law_type=law_type):

            winner = pair.get_winner()
            if winner is None:
                continue

            prompt = render_prompt(pair.original_text, pair.rubric, self.prompt_builder)
            sft_data.append({
                "text": f"{prompt}\n{winner}",
            })

        if not sft_data:
            raise ValueError("No winners available for SFT training after filtering")

        logger.info(f"  Extracted {len(sft_data)} winners for SFT training")

        train_dataset = Dataset.from_list(sft_data)

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA config
        peft_config = None
        if self.config.use_lora:
            try:
                from peft import LoraConfig, TaskType
                peft_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    lora_dropout=self.config.lora_dropout,
                    target_modules=self.config.lora_target_modules,
                    task_type=TaskType.CAUSAL_LM,
                )
            except ImportError:
                logger.warning("peft not available, training without LoRA")

        # SFT config
        training_args = SFTConfig(
            output_dir=str(output_dir),
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            max_seq_length=self.config.max_length,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            bf16=self.config.bf16,
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        # Train
        logger.info("Starting SFT training...")
        trainer.train()

        # Save
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        logger.info(f"SFT training complete. Model saved to {final_path}")

        return str(final_path)


# =============================================================================
# GRPO Trainer
# =============================================================================

@register_trainer("grpo")
class GRPOGeneratorTrainer(BaseGeneratorTrainer):
    """
    TRL GRPO - online generation with reward function.

    GRPO generates k completions per prompt and scores them using a
    reward function. The model is updated to increase probability of
    higher-scoring completions.
    """

    def __init__(
        self,
        genrm_judge: Optional[Any] = None,
        config: Optional[GeneratorTrainerConfig] = None,
        prompt_builder: Optional[Callable[[str, str], Any]] = None,
        num_generations: int = 4,
    ):
        """
        Initialize GRPO trainer.

        Args:
            genrm_judge: Optional GenRM judge for creating reward function
            config: Training configuration
            prompt_builder: Optional prompt builder
            num_generations: Number of generations per prompt
        """
        super().__init__(config, prompt_builder)
        self.genrm_judge = genrm_judge
        self.num_generations = num_generations

    @property
    def method_name(self) -> str:
        return "GRPO"

    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        law_type: Optional[str] = None,
        reward_funcs: Optional[Union[Callable, List[Callable]]] = None,
        **kwargs,
    ) -> str:
        """
        Train using GRPO with online generation.

        Args:
            preferences: SupervisionDataset or binary projection dataset
            model_name: HuggingFace model name to fine-tune
            output_dir: Directory to save trained model
            law_type: Optional filter for specific law type
            reward_funcs: Reward function(s) for GRPO. If None and genrm_judge
                         is set, creates reward function from GenRM.
                         In large-model-only mode, callers should pass reward_funcs.
            **kwargs: Additional arguments

        Returns:
            Path to saved model
        """
        from treepo._research.training.config_sections import OptimizerConfig, RuntimeConfig, TrainConfig
        from treepo._research.training.trl_training import (
            TRLGRPOConfig,
            TRLLoraConfig,
            TRLPropensityWeightingConfig,
            TRLQuantizationConfig,
            TRLSequenceConfig,
            TRLTrainingConfig,
            train_grpo,
        )

        logger.info(f"Starting {self.method_name} training with model: {model_name}")

        # Create reward function from GenRM if not provided
        if reward_funcs is None:
            if self.genrm_judge is not None:
                from treepo._research.training.supervision.rewards import create_genrm_reward_func
                reward_funcs = create_genrm_reward_func(self.genrm_judge)
            else:
                raise ValueError(
                    "GRPO training requires either reward_funcs or genrm_judge. "
                    "TRL GRPOTrainer is online and generates completions at runtime."
                )

        # Create TRL config
        trl_config = TRLTrainingConfig(
            train=TrainConfig(
                epochs=self.config.num_train_epochs,
                batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
            ),
            optimizer=OptimizerConfig(
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
            ),
            runtime=RuntimeConfig(bf16=self.config.bf16, gradient_checkpointing=True),
            sequence=TRLSequenceConfig(max_length=self.config.max_length),
            grpo=TRLGRPOConfig(num_generations=self.num_generations),
            lora=TRLLoraConfig(
                use_lora=self.config.use_lora,
                lora_r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                lora_target_modules=self.config.lora_target_modules,
            ),
            quantization=TRLQuantizationConfig(load_in_4bit=self.config.load_in_4bit),
            propensity_weighting=TRLPropensityWeightingConfig(
                use_propensity_weighting=self.config.use_propensity_weighting,
                propensity_resample=self.config.propensity_resample,
                propensity_native_loss_weighting=self.config.propensity_native_loss_weighting,
                propensity_weight_clip=self.config.propensity_weight_clip,
                propensity_random_seed=self.config.propensity_random_seed,
                propensity_sampling_strategy=self.config.propensity_sampling_strategy,
                propensity_stratify_key=self.config.propensity_stratify_key,
            ),
        )

        return train_grpo(
            dataset=preferences,
            model_name=model_name,
            output_dir=output_dir,
            config=trl_config,
            law_type=law_type,
            reward_funcs=reward_funcs,
            prompt_builder=self.prompt_builder,
        )


# =============================================================================
# BootstrapFinetune Trainer
# =============================================================================

@register_trainer("bootstrap_finetune")
class BootstrapFinetuneTrainer(BaseGeneratorTrainer):
    """
    DSPy BootstrapFinetune - teacher-student distillation.

    Uses DSPy's BootstrapFinetune to distill from a teacher model,
    filtering traces by a metric (GenRM quality) to only learn from
    high-quality examples.
    """

    def __init__(
        self,
        genrm_judge: Optional[Any] = None,
        teacher_lm: Optional[Any] = None,
        config: Optional[GeneratorTrainerConfig] = None,
        prompt_builder: Optional[Callable[[str, str], Any]] = None,
        metric_threshold: float = 3.0,
    ):
        """
        Initialize BootstrapFinetune trainer.

        Args:
            genrm_judge: GenRM judge for creating DSPy metric
            teacher_lm: Teacher language model for distillation
            config: Training configuration
            prompt_builder: Optional prompt builder
            metric_threshold: GenRM ranking threshold for filtering (1-3 = good)
        """
        super().__init__(config, prompt_builder)
        self.genrm_judge = genrm_judge
        self.teacher_lm = teacher_lm
        self.metric_threshold = metric_threshold

    @property
    def method_name(self) -> str:
        return "BootstrapFinetune"

    def train(
        self,
        preferences: Union[BinaryProjectionDataset, SupervisionDataset],
        model_name: str,
        output_dir: Union[str, Path],
        law_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Train using DSPy BootstrapFinetune.

        Args:
            preferences: SupervisionDataset or binary projection dataset
            model_name: Model to fine-tune (student)
            output_dir: Directory to save trained model
            law_type: Optional filter for specific law type
            **kwargs: Additional arguments

        Returns:
            Path to saved model
        """
        try:
            import dspy
        except ImportError:
            raise ImportError("DSPy library required. Install with: pip install dspy")

        logger.info(f"Starting {self.method_name} training with model: {model_name}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define summarizer signature
        class SummarizerSignature(dspy.Signature):
            """Summarize text while preserving specified information."""
            content: str = dspy.InputField(desc="Text to summarize")
            rubric: str = dspy.InputField(desc="What information to preserve")
            summary: str = dspy.OutputField(desc="Summary preserving the rubric information")

        # Setup student model
        student_lm = dspy.LM(model=model_name)
        student = dspy.ChainOfThought(SummarizerSignature)

        # Setup teacher (use provided or default to GPT-4)
        if self.teacher_lm is not None:
            teacher = dspy.ChainOfThought(SummarizerSignature)
            teacher.set_lm(self.teacher_lm)
        else:
            logger.warning("No teacher_lm provided, using student as teacher (self-distillation)")
            teacher = student

        # Build trainset from tournament winners
        trainset = []
        for pair in self._prepare_pairs(preferences, law_type=law_type):

            winner = pair.get_winner()
            if winner is None:
                continue

            example = dspy.Example(
                content=pair.original_text,
                rubric=pair.rubric,
                summary=winner,
            ).with_inputs("content", "rubric")
            trainset.append(example)

        if not trainset:
            raise ValueError("No training examples after filtering")

        logger.info(f"  Built trainset with {len(trainset)} examples")

        # Create metric from GenRM
        from treepo._research.training.supervision.rewards import create_genrm_dspy_metric
        metric = create_genrm_dspy_metric(
            self.genrm_judge,
            threshold=self.metric_threshold,
        )

        # Configure and run BootstrapFinetune
        dspy.settings.experimental = True
        optimizer = dspy.BootstrapFinetune(
            metric=metric,
            num_threads=4,
            train_kwargs={
                "use_peft": self.config.use_lora,
                "num_train_epochs": self.config.num_train_epochs,
            },
        )

        with dspy.context(lm=student_lm):
            optimized_summarizer = optimizer.compile(
                student,
                teacher=teacher,
                trainset=trainset,
            )

        # Save the optimized module
        final_path = output_dir / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        optimized_summarizer.save(str(final_path / "summarizer.json"))

        logger.info(f"BootstrapFinetune training complete. Model saved to {final_path}")

        return str(final_path)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trainer_from_method(
    method: Literal["dpo", "sft", "grpo", "bootstrap_finetune"],
    genrm_judge: Optional[Any] = None,
    teacher_lm: Optional[Any] = None,
    config: Optional[GeneratorTrainerConfig] = None,
    prompt_builder: Optional[Callable] = None,
) -> BaseGeneratorTrainer:
    """
    Create a trainer for the specified method.

    This is a convenience function that handles method-specific initialization.

    Args:
        method: Training method name
        genrm_judge: Optional GenRM judge (legacy convenience for GRPO/BootstrapFinetune)
        teacher_lm: Teacher model (for BootstrapFinetune)
        config: Training configuration
        prompt_builder: Optional prompt builder

    Returns:
        Configured trainer instance
    """
    if method == "grpo":
        return GRPOGeneratorTrainer(
            genrm_judge=genrm_judge,
            config=config,
            prompt_builder=prompt_builder,
        )
    elif method == "bootstrap_finetune":
        return BootstrapFinetuneTrainer(
            genrm_judge=genrm_judge,
            teacher_lm=teacher_lm,
            config=config,
            prompt_builder=prompt_builder,
        )
    else:
        return get_trainer(
            method,
            config=config,
            prompt_builder=prompt_builder,
        )
