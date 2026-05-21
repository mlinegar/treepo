"""
Helpers for resolving vLLM serve/runtime flags from settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(part for part in parts if part)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        out: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    text = str(value).strip()
    if not text:
        return ()
    return (text,)


def _coerce_dict_or_none(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return {str(k): v for k, v in parsed.items()}
    return None


@dataclass
class VLLMRuntimeFlags:
    """
    Runtime flags that tune vLLM serving behavior.

    These are intentionally transport-agnostic and can be converted to CLI args
    for `vllm.entrypoints.openai.api_server`.
    """

    enforce_eager: bool = False
    api_server_count: int = 1
    disable_frontend_multiprocessing: bool = False
    mm_processor_cache_gb: Optional[float] = None
    mm_processor_cache_type: Optional[str] = None
    interleave_mm_strings: bool = False
    allowed_media_domains: Tuple[str, ...] = ()
    limit_mm_per_prompt: Optional[Dict[str, Any]] = None
    extra_flags: Tuple[str, ...] = field(default_factory=tuple)

    def to_cli_args(self) -> List[str]:
        """Render runtime flags as CLI args."""
        args: List[str] = []

        if self.enforce_eager:
            args.append("--enforce-eager")

        if self.api_server_count > 1:
            args.extend(["--api-server-count", str(int(self.api_server_count))])

        if self.disable_frontend_multiprocessing:
            args.append("--disable-frontend-multiprocessing")

        if self.mm_processor_cache_gb is not None:
            args.extend(["--mm-processor-cache-gb", f"{float(self.mm_processor_cache_gb):g}"])

        if self.mm_processor_cache_type:
            args.extend(["--mm-processor-cache-type", str(self.mm_processor_cache_type)])

        if self.interleave_mm_strings:
            args.append("--interleave-mm-strings")

        if self.allowed_media_domains:
            domains_json = json.dumps(list(self.allowed_media_domains), separators=(",", ":"))
            args.extend(["--allowed-media-domains", domains_json])

        if self.limit_mm_per_prompt:
            limit_json = json.dumps(self.limit_mm_per_prompt, separators=(",", ":"))
            args.extend(["--limit-mm-per-prompt", limit_json])

        if self.extra_flags:
            args.extend(list(self.extra_flags))

        return args


def resolve_vllm_runtime_flags(vllm_cfg: Mapping[str, Any], profile: str) -> VLLMRuntimeFlags:
    """
    Resolve vLLM runtime flags for a model profile.

    Expected config shape:
      vllm:
        runtime:
          enforce_eager_default: false
          enforce_eager_profiles: [glm-4.6]
          api_server_count: 1
          profile_overrides:
            qwen-vl-235b:
              limit_mm_per_prompt: {image: 8, video: 2}
    """
    runtime_cfg_raw = vllm_cfg.get("runtime", {})
    runtime_cfg = dict(runtime_cfg_raw) if isinstance(runtime_cfg_raw, Mapping) else {}

    profile_overrides_raw = runtime_cfg.get("profile_overrides", {})
    profile_overrides = (
        profile_overrides_raw if isinstance(profile_overrides_raw, Mapping) else {}
    )
    profile_cfg_raw = profile_overrides.get(profile, {})
    profile_cfg = dict(profile_cfg_raw) if isinstance(profile_cfg_raw, Mapping) else {}

    merged_cfg: Dict[str, Any] = dict(runtime_cfg)
    merged_cfg.pop("profile_overrides", None)
    merged_cfg.update(profile_cfg)

    extra_flags_default = _coerce_str_list(runtime_cfg.get("extra_flags"))
    extra_flags_profile = _coerce_str_list(profile_cfg.get("extra_flags"))
    extra_flags = extra_flags_default + extra_flags_profile

    explicit_eager = merged_cfg.get("enforce_eager")
    if explicit_eager is not None:
        enforce_eager = _coerce_bool(explicit_eager, default=False)
    else:
        eager_default = _coerce_bool(runtime_cfg.get("enforce_eager_default"), default=False)
        eager_profiles = set(_coerce_str_list(runtime_cfg.get("enforce_eager_profiles")))
        enforce_eager = eager_default or profile in eager_profiles

    raw_limit_default = _coerce_dict_or_none(runtime_cfg.get("limit_mm_per_prompt"))
    raw_limit_profile = _coerce_dict_or_none(profile_cfg.get("limit_mm_per_prompt"))
    if raw_limit_default is not None and raw_limit_profile is not None:
        limit_mm_per_prompt: Optional[Dict[str, Any]] = dict(raw_limit_default)
        limit_mm_per_prompt.update(raw_limit_profile)
    elif raw_limit_profile is not None:
        limit_mm_per_prompt = raw_limit_profile
    else:
        limit_mm_per_prompt = raw_limit_default

    mm_processor_cache_gb = None
    if "mm_processor_cache_gb" in merged_cfg:
        mm_processor_cache_gb = _coerce_float_or_none(merged_cfg.get("mm_processor_cache_gb"))

    mm_processor_cache_type = merged_cfg.get("mm_processor_cache_type")
    if mm_processor_cache_type is not None:
        mm_processor_cache_type = str(mm_processor_cache_type).strip() or None

    return VLLMRuntimeFlags(
        enforce_eager=enforce_eager,
        api_server_count=max(1, _coerce_int(merged_cfg.get("api_server_count"), default=1)),
        disable_frontend_multiprocessing=_coerce_bool(
            merged_cfg.get("disable_frontend_multiprocessing"),
            default=False,
        ),
        mm_processor_cache_gb=mm_processor_cache_gb,
        mm_processor_cache_type=mm_processor_cache_type,
        interleave_mm_strings=_coerce_bool(
            merged_cfg.get("interleave_mm_strings"),
            default=False,
        ),
        allowed_media_domains=_coerce_str_list(merged_cfg.get("allowed_media_domains")),
        limit_mm_per_prompt=limit_mm_per_prompt,
        extra_flags=extra_flags,
    )
