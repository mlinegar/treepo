"""
Data loader for Manifesto Project data.

Loads manifestos with text and RILE scores for OPS evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Iterator, Tuple
import pandas as pd
import json
from datetime import datetime


@dataclass
class ManifestoSample:
    """A single manifesto sample with text and ground truth score."""

    manifesto_id: str           # Format: "{party}_{date}" e.g., "51320_196410"
    party_id: int               # CMP party code
    party_name: str             # Full party name
    party_abbrev: str           # Party abbreviation
    country_code: int           # CMP country code
    country_name: str           # Country name
    election_date: str          # Election date (YYYY-MM-DD format)
    date_code: int              # Date code (YYYYMM format)
    text: str                   # Full manifesto text
    rile: float                 # RILE score (-100 to +100)

    # Optional metadata
    vote_share: Optional[float] = None
    party_family: Optional[int] = None  # Party family code (parfam)

    @property
    def year(self) -> int:
        """Extract year from date code."""
        return self.date_code // 100

    def __repr__(self) -> str:
        return (
            f"ManifestoSample(id={self.manifesto_id}, "
            f"party={self.party_abbrev}, country={self.country_name}, "
            f"year={self.year}, rile={self.rile:.1f})"
        )


# Western European country codes (CMP codes)
WESTERN_EUROPE_CODES = {
    11: "Sweden",
    12: "Norway",
    13: "Denmark",
    14: "Finland",
    21: "Iceland",
    22: "Netherlands",
    23: "Belgium",
    31: "France",
    32: "Italy",
    33: "Spain",
    34: "Greece",
    35: "Portugal",
    41: "Germany",
    42: "Austria",
    43: "Switzerland",
    51: "United Kingdom",
    53: "Ireland",
    54: "Luxembourg",
}

# Party family codes
PARTY_FAMILIES = {
    10: "Ecological",
    20: "Communist/Socialist",
    30: "Social Democratic",
    40: "Liberal",
    50: "Christian Democratic",
    60: "Conservative",
    70: "Nationalist",
    80: "Agrarian",
    90: "Ethnic/Regional",
    95: "Special Issue",
    98: "Independents",
}


class ManifestoDataset:
    """
    Dataset for Manifesto Project data.

    Loads manifestos from the main dataset CSV and text files,
    filters by criteria, and provides iteration over samples.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        countries: Optional[List[int]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
        require_text: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to a Manifesto corpus directory with
                manifesto_maindataset.csv and texts/.
            countries: List of country codes to include (None = all)
            min_year: Minimum election year to include
            max_year: Maximum election year to include
            require_text: Only include manifestos with text files
        """
        if data_dir is None:
            # Prefer the fetched full-document corpus. The legacy
            # manifesto_project_full/texts surface is a fragment corpus and is
            # quarantined under manifesto_project_full_OLD/texts.
            root = Path(__file__).parent.parent.parent.parent
            full_doc_dir = root / "data" / "raw" / "manifesto_corpus_benoit"
            legacy_dir = root / "data" / "raw" / "manifesto_project_full"
            data_dir = full_doc_dir if full_doc_dir.exists() else legacy_dir

        self.data_dir = Path(data_dir)
        self.countries = countries
        self.min_year = min_year
        self.max_year = max_year
        self.require_text = require_text

        # Paths
        self.csv_path = self.data_dir / "manifesto_maindataset.csv"
        self.texts_dir = self.data_dir / "texts"

        # Load and filter metadata
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load the main dataset CSV and apply filters."""
        # Load CSV
        self.metadata_df = pd.read_csv(self.csv_path, low_memory=False)

        # Create manifesto_id column
        self.metadata_df["manifesto_id"] = (
            self.metadata_df["party"].astype(str) + "_" +
            self.metadata_df["date"].astype(str)
        )

        # Extract year from date
        self.metadata_df["year"] = self.metadata_df["date"] // 100

        # Apply filters
        mask = pd.Series([True] * len(self.metadata_df))

        if self.countries is not None:
            mask &= self.metadata_df["country"].isin(self.countries)

        if self.min_year is not None:
            mask &= self.metadata_df["year"] >= self.min_year

        if self.max_year is not None:
            mask &= self.metadata_df["year"] <= self.max_year

        # Filter to only manifestos with RILE scores
        mask &= self.metadata_df["rile"].notna()

        # Filter to manifestos with text files if required
        if self.require_text:
            has_text = self.metadata_df["manifesto_id"].apply(self._has_text_file)
            mask &= has_text

        self.filtered_df = self.metadata_df[mask].copy()

    def _has_text_file(self, manifesto_id: str) -> bool:
        """Check if a text file exists for this manifesto."""
        text_path = self.texts_dir / f"{manifesto_id}.txt"
        return text_path.exists()

    def _load_text(self, manifesto_id: str) -> Optional[str]:
        """Load text content for a manifesto."""
        text_path = self.texts_dir / f"{manifesto_id}.txt"
        if text_path.exists():
            return text_path.read_text(encoding="utf-8")
        return None

    def get_sample(self, manifesto_id: str) -> Optional[ManifestoSample]:
        """
        Load a single manifesto sample by ID.

        Args:
            manifesto_id: The manifesto ID (e.g., "51320_196410")

        Returns:
            ManifestoSample or None if not found
        """
        row = self.filtered_df[self.filtered_df["manifesto_id"] == manifesto_id]
        if len(row) == 0:
            return None

        row = row.iloc[0]
        text = self._load_text(manifesto_id)

        if text is None and self.require_text:
            return None

        return ManifestoSample(
            manifesto_id=manifesto_id,
            party_id=int(row["party"]),
            party_name=row["partyname"],
            party_abbrev=row.get("partyabbrev", "") or "",
            country_code=int(row["country"]),
            country_name=row["countryname"],
            election_date=str(row.get("edate", "")),
            date_code=int(row["date"]),
            text=text or "",
            rile=float(row["rile"]),
            vote_share=float(row["pervote"]) if pd.notna(row.get("pervote")) else None,
            party_family=int(row["parfam"]) if pd.notna(row.get("parfam")) else None,
        )

    def get_all_ids(self) -> List[str]:
        """Get all manifesto IDs in the filtered dataset."""
        return self.filtered_df["manifesto_id"].tolist()

    def __len__(self) -> int:
        """Number of manifestos in filtered dataset."""
        return len(self.filtered_df)

    def __iter__(self) -> Iterator[ManifestoSample]:
        """Iterate over all samples in the dataset."""
        for manifesto_id in self.get_all_ids():
            sample = self.get_sample(manifesto_id)
            if sample is not None:
                yield sample

    def get_stats(self) -> Dict:
        """Get statistics about the dataset."""
        df = self.filtered_df
        return {
            "total_manifestos": int(len(df)),
            "countries": int(df["countryname"].nunique()),
            "country_list": df["countryname"].unique().tolist(),
            "parties": int(df["party"].nunique()),
            "year_range": (int(df["year"].min()), int(df["year"].max())),
            "rile_range": (float(df["rile"].min()), float(df["rile"].max())),
            "rile_mean": float(df["rile"].mean()),
            "rile_std": float(df["rile"].std()),
        }

    def create_temporal_split(
        self,
        train_end_year: int = 1995,
        val_end_year: int = 2005,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Create temporal train/val/test split.

        Args:
            train_end_year: Last year for training set (inclusive)
            val_end_year: Last year for validation set (inclusive)

        Returns:
            Tuple of (train_ids, val_ids, test_ids)
        """
        df = self.filtered_df

        train_mask = df["year"] <= train_end_year
        val_mask = (df["year"] > train_end_year) & (df["year"] <= val_end_year)
        test_mask = df["year"] > val_end_year

        train_ids = df[train_mask]["manifesto_id"].tolist()
        val_ids = df[val_mask]["manifesto_id"].tolist()
        test_ids = df[test_mask]["manifesto_id"].tolist()

        return train_ids, val_ids, test_ids

    def get_split_samples(
        self,
        ids: List[str]
    ) -> Iterator[ManifestoSample]:
        """Iterate over samples for given IDs."""
        for manifesto_id in ids:
            sample = self.get_sample(manifesto_id)
            if sample is not None:
                yield sample


def create_pilot_dataset(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    train_end_year: int = 1995,
    val_end_year: int = 2005,
) -> Dict:
    """
    Create a pilot dataset focused on Western European countries.

    Args:
        data_dir: Path to manifesto data
        output_dir: Where to save the split files
        train_end_year: Last year for training
        val_end_year: Last year for validation

    Returns:
        Dictionary with split statistics
    """
    # Focus countries: UK, Germany, France, Sweden, Netherlands
    pilot_countries = [51, 41, 31, 11, 22]

    dataset = ManifestoDataset(
        data_dir=data_dir,
        countries=pilot_countries,
        require_text=True,
    )

    train_ids, val_ids, test_ids = dataset.create_temporal_split(
        train_end_year=train_end_year,
        val_end_year=val_end_year,
    )

    # Save splits if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "train_ids.json", "w") as f:
            json.dump(train_ids, f, indent=2)

        with open(output_dir / "val_ids.json", "w") as f:
            json.dump(val_ids, f, indent=2)

        with open(output_dir / "test_ids.json", "w") as f:
            json.dump(test_ids, f, indent=2)

        # Save metadata
        metadata = {
            "countries": {code: WESTERN_EUROPE_CODES.get(code, "Unknown") for code in pilot_countries},
            "train_end_year": train_end_year,
            "val_end_year": val_end_year,
            "split_sizes": {
                "train": len(train_ids),
                "val": len(val_ids),
                "test": len(test_ids),
            },
            "total": len(train_ids) + len(val_ids) + len(test_ids),
            "dataset_stats": dataset.get_stats(),
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "dataset": dataset,
    }


if __name__ == "__main__":
    # Test the data loader
    print("Testing ManifestoDataset...")

    # Load pilot dataset
    result = create_pilot_dataset()

    print(f"\nPilot dataset created:")
    print(f"  Train: {len(result['train_ids'])} manifestos")
    print(f"  Val: {len(result['val_ids'])} manifestos")
    print(f"  Test: {len(result['test_ids'])} manifestos")

    print(f"\nDataset stats:")
    for key, value in result['dataset'].get_stats().items():
        print(f"  {key}: {value}")

    # Load a sample
    if result['train_ids']:
        sample_id = result['train_ids'][0]
        sample = result['dataset'].get_sample(sample_id)
        if sample:
            print(f"\nSample manifesto:")
            print(f"  {sample}")
            print(f"  Text length: {len(sample.text)} chars")
