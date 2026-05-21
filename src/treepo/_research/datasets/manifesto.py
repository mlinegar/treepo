"""
Manifesto dataset plugin.

The Manifesto dataset is the default data source for OPS training.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import random

from treepo._research.core.documents import DocumentSample
from treepo._research.tasks.manifesto.data_loader import ManifestoDataset as DocDataset, ManifestoSample

from .base import DatasetInfo, register_dataset


@register_dataset("manifesto")
class ManifestoDataset:
    """Dataset plugin for Manifesto Project data."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        countries: Optional[List[int]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        require_text: bool = True,
    ):
        self._dataset = DocDataset(
            data_dir=data_dir,
            countries=countries,
            min_year=min_year,
            max_year=max_year,
            require_text=require_text,
        )

    @property
    def name(self) -> str:
        return "manifesto"

    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.name,
            description="Manifesto Project dataset with RILE scores",
            supports_reference_scores=True,
        )

    def load_samples(
        self,
        limit: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        **kwargs: Any,
    ) -> List[DocumentSample]:
        samples = list(self._dataset)

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)

        if limit is not None:
            samples = samples[:limit]

        return [
            DocumentSample(
                doc_id=sample.manifesto_id,
                text=sample.text,
                reference_score=sample.rile,
                metadata={
                    "party_id": sample.party_id,
                    "party_name": sample.party_name,
                    "party_abbrev": sample.party_abbrev,
                    "country_code": sample.country_code,
                    "country_name": sample.country_name,
                    "year": sample.year,
                    "election_date": sample.election_date,
                    "date_code": sample.date_code,
                    "vote_share": sample.vote_share,
                    "party_family": sample.party_family,
                },
            )
            for sample in samples
        ]
