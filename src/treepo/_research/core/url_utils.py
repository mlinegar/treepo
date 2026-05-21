"""URL normalization helpers shared across engine surfaces."""

from __future__ import annotations


def normalize_generate_base_url(
    base_url: str,
    *,
    generate_path: str = "/generate",
    openai_base_path: str = "/v1",
) -> str:
    """Normalize a `/generate` base URL so callers don't double-append paths.

    Contract:
    - The backend stores `base_url` as the host:port (optionally with a prefix path)
    - The request URL is `base_url + generate_path`
    - If the provided base_url already ends with generate_path, strip it.
    - If the provided base_url ends with an OpenAI-compatible base path (default: `/v1`),
      strip it as well. This supports generate-first usage when callers only know the
      OpenAI base URL but want to hit `/generate`.
    """
    rendered = str(base_url or "").rstrip("/")
    suffix = str(generate_path or "").strip() or "/generate"
    if not suffix.startswith("/"):
        suffix = f"/{suffix}"
    if rendered.endswith(suffix):
        rendered = rendered[: -len(suffix)].rstrip("/")
    openai_suffix = str(openai_base_path or "").strip() or "/v1"
    if not openai_suffix.startswith("/"):
        openai_suffix = f"/{openai_suffix}"
    if rendered.endswith(openai_suffix):
        rendered = rendered[: -len(openai_suffix)].rstrip("/")
    return rendered


__all__ = ["normalize_generate_base_url"]
