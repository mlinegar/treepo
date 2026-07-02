"""Bit-hashing feature encoder for numeric-token leaf sequences.

Extracted from :mod:`treepo.methods.fno` to keep that module readable. The
public entry point is :func:`add_numeric_sequence_features`; the remaining
functions are internal to the hashing scheme.

``np`` is threaded through as a parameter rather than imported at module load
so the FNO family stays import-light (numpy is only touched on the numeric
encoding path).
"""

from __future__ import annotations

from typing import Any, Sequence


def add_numeric_sequence_features(row: Any, tokens: Sequence[int], *, dim: int, np: Any) -> None:
    """Accumulate hashed features for one numeric-token leaf into ``row``.

    ``row`` is a length-``dim`` float buffer that is mutated in place. The
    scheme is a feature-hashing (hashing-trick) encoder over the integer token
    sequence, blending several complementary views so a downstream linear /
    conv operator can recover changepoint-style structure:

    - A few low-index slots are *reserved* (see
      :func:`_numeric_hash_reserved_slots`) for dense, un-hashed summaries:
      slot 0 holds ``sqrt(n)`` (a length anchor), and slots 1..3 hold the
      number of value changes at successively coarser bit granularities.
    - The remaining slots hold hashed *counts*. Raw tokens plus three
      bit-shifted ("coarsened") copies are each run through a 64-bit
      finalizer-style mix (``_hash_numeric_array``), binned into the
      non-reserved slot range, and added with per-view weights so fine and
      coarse structure both contribute.
    - Positional and transition cues are hashed in separately: the first and
      last coarse token get extra weight per granularity, and, when a coarse
      value differs from its predecessor, the ordered ``(prev, cur)`` pair is
      hashed so adjacent-value transitions leave a signature.

    Distinct 64-bit ``salt`` constants per view keep the hashed sub-features
    from colliding systematically. All arithmetic uses unsigned 64-bit numpy
    integers to match the C-style mixing constants.
    """
    token_arr = np.asarray(tokens, dtype=np.uint64)
    if token_arr.size == 0:
        return
    n = int(token_arr.size)
    _add_reserved_numeric_feature(row, 0, float(np.sqrt(max(1, n))))
    _add_hashed_counts(row, _hash_numeric_array(token_arr, dim=dim, salt=0x9E3779B185EBCA87, np=np), weight=1.0, np=np)
    for shift, weight in ((4, 0.5), (8, 0.75), (12, 0.35)):
        coarse = np.right_shift(token_arr, np.uint64(shift))
        _add_hashed_counts(
            row,
            _hash_numeric_array(coarse, dim=dim, salt=0xC2B2AE3D27D4EB4F + shift, np=np),
            weight=weight,
            np=np,
        )
    length_bucket = _hash_numeric_value(n, dim=dim, salt=0xA24BAED4963EE407)
    row[length_bucket] += float(np.sqrt(max(1, n)))
    for slot_idx, shift in enumerate((4, 8, 12), start=1):
        coarse = np.right_shift(token_arr, np.uint64(shift))
        first_bucket = _hash_numeric_value(int(coarse[0]), dim=dim, salt=0x165667B19E3779F9 + shift)
        last_bucket = _hash_numeric_value(int(coarse[-1]), dim=dim, salt=0x85EBCA77C2B2AE63 + shift)
        row[first_bucket] += 2.0
        row[last_bucket] += 2.0
        if n > 1:
            changes = int(np.count_nonzero(coarse[1:] != coarse[:-1]))
            _add_reserved_numeric_feature(row, slot_idx, float(changes))
            change_bucket = _hash_numeric_value(shift, dim=dim, salt=0x27D4EB2F165667C5)
            row[change_bucket] += float(changes)
            pair_values = (coarse[:-1] << np.uint64(32)) ^ coarse[1:]
            changed_pairs = pair_values[coarse[1:] != coarse[:-1]]
            if changed_pairs.size:
                _add_hashed_counts(
                    row,
                    _hash_numeric_array(changed_pairs, dim=dim, salt=0x94D049BB133111EB + shift, np=np),
                    weight=0.25,
                    np=np,
                )


def _add_reserved_numeric_feature(row: Any, index: int, value: float) -> None:
    if 0 <= int(index) < int(row.shape[0]):
        row[int(index)] += float(value)


def _add_hashed_counts(row: Any, buckets: Any, *, weight: float, np: Any) -> None:
    counts = np.bincount(buckets.astype(np.int64), minlength=int(row.shape[0])).astype(np.float32)
    row += counts[: int(row.shape[0])] * float(weight)


def _hash_numeric_array(values: Any, *, dim: int, salt: int, np: Any) -> Any:
    arr = np.asarray(values, dtype=np.uint64) ^ np.uint64(salt)
    arr ^= arr >> np.uint64(33)
    arr *= np.uint64(0xFF51AFD7ED558CCD)
    arr ^= arr >> np.uint64(33)
    arr *= np.uint64(0xC4CEB9FE1A85EC53)
    arr ^= arr >> np.uint64(33)
    dim = max(1, int(dim))
    reserved = _numeric_hash_reserved_slots(dim)
    width = max(1, dim - reserved)
    out = np.mod(arr, np.uint64(width)).astype(np.int64)
    if dim > reserved:
        out += int(reserved)
    return out


def _hash_numeric_value(value: int, *, dim: int, salt: int) -> int:
    dim = max(1, int(dim))
    x = (int(value) ^ int(salt)) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xFF51AFD7ED558CCD) & ((1 << 64) - 1)
    x ^= x >> 33
    x = (x * 0xC4CEB9FE1A85EC53) & ((1 << 64) - 1)
    x ^= x >> 33
    reserved = _numeric_hash_reserved_slots(dim)
    width = max(1, dim - reserved)
    out = int(x % width)
    return out + reserved if dim > reserved else int(x % dim)


def _numeric_hash_reserved_slots(dim: int) -> int:
    dim = max(1, int(dim))
    return min(8, dim) if dim >= 16 else min(4, dim)
