from __future__ import annotations

import random
from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

from treepo._research.core.logged_supervision import ObservationUnitKind, SamplingMetadata
from treepo._research.core.preference_supervision import preference_supervision_metadata

from .preferences import derive_preference_from_utilities
from .protocols import CandidateGenerator, Compressor
from .records import PairwisePreference

X = TypeVar("X")
Z = TypeVar("Z")
A = TypeVar("A")
Y = TypeVar("Y")


def collect_pairwise_preferences(
    examples: Iterable[X],
    *,
    candidate_generator: CandidateGenerator[X, A],
    utility_fn: Callable[[X, A], float],
    example_id_fn: Optional[Callable[[X, int], str]] = None,
    rubric: str = "",
    tie_margin: float = 0.0,
    sampling_fn: Optional[Callable[[X], SamplingMetadata]] = None,
    n_pairs_per_example: int = 1,
    seed: int = 0,
) -> List[PairwisePreference]:
    """Collect pairwise preferences by sampling candidates and comparing utilities.

    This is the most general bridge between *any* scalar utility and preference
    learning objectives (DPO/GRPO/etc.): the preferred candidate is the one with
    higher utility. When utilities are oracle-defined, this corresponds to
    "oracle-induced preferences".
    """
    id_fn = example_id_fn or (lambda _x, i: str(i))
    rng = random.Random(int(seed))
    records: List[PairwisePreference] = []

    for idx, example in enumerate(examples):
        example_id = id_fn(example, idx)
        sampling = (
            sampling_fn(example)
            if sampling_fn is not None
            else SamplingMetadata(unit_kind=ObservationUnitKind.PAIR)
        )

        for _ in range(int(n_pairs_per_example)):
            # Derive a deterministic-ish seed per pair so generators can be reproducible.
            pair_seed = rng.randrange(1 << 63)
            candidates = list(candidate_generator.generate(example, n=2, seed=pair_seed))
            if len(candidates) < 2:
                raise ValueError(
                    f"candidate_generator returned {len(candidates)} candidates; expected >= 2"
                )
            cand_a, cand_b = candidates[0], candidates[1]
            utility_a = float(utility_fn(example, cand_a))
            utility_b = float(utility_fn(example, cand_b))
            outcome = derive_preference_from_utilities(
                utility_a,
                utility_b,
                tie_margin=tie_margin,
            )
            records.append(
                PairwisePreference(
                    example_id=example_id,
                    input=example,
                    rubric=rubric,
                    candidate_a=cand_a,
                    candidate_b=cand_b,
                    preferred=outcome.preferred,
                    confidence=outcome.confidence,
                    reasoning=outcome.reasoning,
                    reference=None,
                    score_a=utility_a,
                    score_b=utility_b,
                    sampling=sampling,
                    preference_supervision=preference_supervision_metadata(
                        application_name="ctreepo_opt_collect"
                    ),
                )
            )

    return records


def collect_proxy_training_data(
    examples: Iterable[X],
    *,
    compressor: Compressor[X, Z],
    oracle: Callable[[X], Y],
    sampling_fn: Optional[Callable[[X], SamplingMetadata]] = None,
) -> tuple[List[Z], List[Y], Optional[List[float]]]:
    """Collect (compressed_input, oracle_target, sample_weight) triples for proxy training."""
    inputs: List[Z] = []
    targets: List[Y] = []
    weights: List[float] = []
    use_weights = sampling_fn is not None

    for example in examples:
        inputs.append(compressor.compress(example))
        targets.append(oracle(example))
        if use_weights:
            sampling = sampling_fn(example)
            weights.append(sampling.ipw_weight())

    return inputs, targets, (weights if use_weights else None)
