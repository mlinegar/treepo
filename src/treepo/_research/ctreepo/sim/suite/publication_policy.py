from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class PublicationCtreepoPolicy:
    profile: str
    seeds: Tuple[int, ...]
    q_rates: Tuple[float, ...]
    q_rates_upper: Tuple[float, ...]
    train_docs_lda: Tuple[int, ...]
    train_docs_hard: Tuple[int, ...]
    train_docs_hard_upper: Tuple[int, ...]
    leaf_tokens_lda: Tuple[int, ...]
    leaf_tokens_hard: Tuple[int, ...]
    cal_rates_lda: Tuple[float, ...]
    cal_rates_hard: Tuple[float, ...]
    cal_rates_upper: Tuple[float, ...]
    n_books_test_lda: int
    n_books_test_hard: int
    doc_tokens_lda: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_publication_ctreepo_policy(profile: str) -> PublicationCtreepoPolicy:
    profile_name = str(profile).strip().lower() or "publication"
    if profile_name == "smoke":
        return PublicationCtreepoPolicy(
            profile="smoke",
            seeds=(0,),
            q_rates=(0.0,),
            q_rates_upper=(0.0,),
            train_docs_lda=(128,),
            train_docs_hard=(128,),
            train_docs_hard_upper=(128,),
            leaf_tokens_lda=(32,),
            leaf_tokens_hard=(16,),
            cal_rates_lda=(0.1,),
            cal_rates_hard=(0.1,),
            cal_rates_upper=(0.1,),
            n_books_test_lda=32,
            n_books_test_hard=32,
            doc_tokens_lda=256,
        )
    if profile_name == "publication":
        return PublicationCtreepoPolicy(
            profile="publication",
            seeds=(0, 1, 2, 3, 4, 5, 6, 7),
            q_rates=(0.0, 0.25, 0.5),
            q_rates_upper=(0.0, 0.25),
            train_docs_lda=(128, 256, 512, 1024, 2048, 4096),
            train_docs_hard=(128, 256, 512, 1024, 2048),
            train_docs_hard_upper=(1024, 2048, 4096),
            leaf_tokens_lda=(32, 16, 8),
            leaf_tokens_hard=(16, 8),
            cal_rates_lda=(0.0, 0.05, 0.1),
            cal_rates_hard=(0.05, 0.1, 0.2),
            cal_rates_upper=(0.1,),
            n_books_test_lda=4000,
            n_books_test_hard=5000,
            doc_tokens_lda=2048,
        )
    raise ValueError(f"unknown publication profile: {profile}")


__all__ = ["PublicationCtreepoPolicy", "resolve_publication_ctreepo_policy"]
