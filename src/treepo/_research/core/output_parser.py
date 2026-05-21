"""
Output parsing utilities with case-insensitive key normalization.

This module handles the common issue where LLMs output JSON with inconsistent
key casing (e.g., 'Score_Value' vs 'score_value'). It provides utilities to
normalize keys and extract values robustly.

Usage:
    from treepo._research.core.output_parser import normalize_output_keys, get_field

    # Normalize a DSPy result or dict
    normalized = normalize_output_keys(result, expected_fields=['score', 'reasoning'])

    # Or get a specific field with fallback
    score = get_field(result, 'score', default=0.0)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def normalize_key(key: str) -> str:
    """
    Normalize a key to lowercase with underscores.

    Examples:
        'Score_Value' -> 'score_value'
        'scoreValue' -> 'scorevalue'
        'ScoreValue' -> 'scorevalue'
        'score-value' -> 'score_value'
    """
    # Replace hyphens with underscores
    key = key.replace('-', '_')
    # Convert to lowercase
    return key.lower()


def find_matching_key(
    obj: Any,
    target_key: str,
    strict: bool = False
) -> Optional[str]:
    """
    Find a key in an object that matches the target key (case-insensitive).

    Args:
        obj: Object to search (dict-like or has attributes)
        target_key: The expected key name
        strict: If True, only return exact matches

    Returns:
        The actual key that matches, or None if not found
    """
    target_normalized = normalize_key(target_key)

    # Get available keys. Prefer mapping-style keys() when available because
    # DSPy Prediction objects expose user fields there while __dict__ holds
    # internal storage metadata only.
    keys: List[str] = []
    seen_keys = set()

    def _append_keys(candidate_keys: Any) -> None:
        for key in candidate_keys:
            key_str = str(key)
            if key_str in seen_keys:
                continue
            seen_keys.add(key_str)
            keys.append(key_str)

    if hasattr(obj, 'keys'):
        try:
            _append_keys(obj.keys())
        except Exception:
            pass

    if hasattr(obj, '__dict__'):
        try:
            _append_keys(vars(obj).keys())
        except Exception:
            pass

    store = getattr(obj, '_store', None)
    if hasattr(store, 'keys'):
        try:
            _append_keys(store.keys())
        except Exception:
            pass

    if not keys:
        return None

    # First try exact match
    if target_key in keys:
        return target_key

    if strict:
        return None

    # Try case-insensitive match
    for key in keys:
        if normalize_key(key) == target_normalized:
            return key

    return None


def get_field(
    obj: Any,
    field_name: str,
    default: Any = None,
    strict: bool = False
) -> Any:
    """
    Get a field from an object with case-insensitive key matching.

    Args:
        obj: Object to get field from (dict-like or has attributes)
        field_name: Expected field name
        default: Default value if field not found
        strict: If True, only accept exact key matches

    Returns:
        Field value or default
    """
    matching_key = find_matching_key(obj, field_name, strict=strict)

    if matching_key is None:
        return default

    # Get the value
    if hasattr(obj, '__getitem__'):
        try:
            return obj[matching_key]
        except (KeyError, TypeError):
            pass

    if hasattr(obj, matching_key):
        return getattr(obj, matching_key)

    # Some prediction-like objects keep user-facing outputs in _store.
    store = getattr(obj, '_store', None)
    if hasattr(store, '__getitem__'):
        try:
            return store[matching_key]
        except (KeyError, TypeError):
            pass

    if hasattr(store, matching_key):
        return getattr(store, matching_key)

    return default


def normalize_output_keys(
    obj: Any,
    expected_fields: List[str],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Normalize an output object to a dict with expected field names.

    This handles the case where LLMs output JSON with varying key casing.
    Maps actual keys to expected keys using case-insensitive matching.

    Args:
        obj: Object to normalize (DSPy result, dict, or object with attributes)
        expected_fields: List of expected field names (canonical casing)
        strict: If True, only include exact matches

    Returns:
        Dict with normalized keys matching expected_fields

    Example:
        >>> result = {'Score_Value': 5.0, 'Reasoning': 'test'}
        >>> normalize_output_keys(result, ['score_value', 'reasoning'])
        {'score_value': 5.0, 'reasoning': 'test'}
    """
    normalized = {}
    missing_fields = []

    for expected_field in expected_fields:
        matching_key = find_matching_key(obj, expected_field, strict=strict)

        if matching_key is not None:
            value = get_field(obj, matching_key, strict=True)
            normalized[expected_field] = value
        else:
            missing_fields.append(expected_field)

    if missing_fields:
        logger.debug(f"Fields not found after normalization: {missing_fields}")

    return normalized


def safe_parse_score(
    obj: Any,
    field_names: List[str],
    default: float = 0.0,
    scale_min: Optional[float] = None,
    scale_max: Optional[float] = None
) -> float:
    """
    Safely extract a numeric score from an object, trying multiple field names.

    Args:
        obj: Object to extract score from
        field_names: List of possible field names (tried in order)
        default: Default value if no field found
        scale_min: Optional minimum value for clamping
        scale_max: Optional maximum value for clamping

    Returns:
        Parsed float score, clamped to valid range
    """
    for field_name in field_names:
        value = get_field(obj, field_name)
        if value is not None:
            try:
                score = float(value)
                if scale_min is not None:
                    score = max(scale_min, score)
                if scale_max is not None:
                    score = min(scale_max, score)
                return score
            except (ValueError, TypeError):
                logger.warning(f"Could not parse {field_name}={value} as float")
                continue

    logger.warning(f"No valid score found in fields {field_names}, using default {default}")
    return default


class NormalizedOutputAccessor:
    """
    Wrapper that provides case-insensitive access to output fields.

    Useful for wrapping DSPy results to handle key casing variations.

    Example:
        result = some_dspy_module(text=text)
        accessor = NormalizedOutputAccessor(result)
        score = accessor.score_value  # Works even if result has 'Score_Value'
    """

    def __init__(self, obj: Any):
        self._obj = obj
        self._cache: Dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)

        if name in self._cache:
            return self._cache[name]

        value = get_field(self._obj, name)
        if value is not None:
            self._cache[name] = value
            return value

        raise AttributeError(f"No field matching '{name}' found in output")

    def get(self, name: str, default: Any = None) -> Any:
        """Get a field with a default value."""
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    def to_dict(self, expected_fields: List[str]) -> Dict[str, Any]:
        """Convert to dict with expected field names."""
        return normalize_output_keys(self._obj, expected_fields)
