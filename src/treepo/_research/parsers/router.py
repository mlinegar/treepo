"""
Parser-feedback routing stage.

Routes parser-emitted action hints (e.g. OCR/VLM/vision-embedding) through a
processor registry and records per-sample execution metadata.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import threading
import time
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

PARSER_ROUTER_CONTRACT_VERSION = 1
PARSER_ROUTER_REQUEST_TYPE = "thinkingtrees.parser_router.request"
PARSER_ROUTER_RESPONSE_TYPE = "thinkingtrees.parser_router.response"

DEFAULT_ROUTER_ACTIONS: Tuple[str, ...] = ("ocr", "vlm_parse", "vision_embedding", "vlm_segment")
_SAMPLE_TEXT_UPDATE_LOCK = threading.Lock()

_ACTION_ALIASES: Dict[str, str] = {
    "ocr": "ocr",
    "vlm": "vlm_parse",
    "vlm_parse": "vlm_parse",
    "vlm_segment": "vlm_segment",
    "visual_segment": "vlm_segment",
    "vision_embed": "vision_embedding",
    "vision_embedding": "vision_embedding",
    "image_embedding": "vision_embedding",
}

_ACTION_BY_ROUTE_NAME: Dict[str, Tuple[str, ...]] = {
    "ocr_first_then_vision_embedding": ("ocr", "vision_embedding"),
    "vlm_parse": ("vlm_parse",),
    "vlm_segment": ("vlm_segment",),
    "augment_with_vision_embedding": ("vision_embedding",),
}


def normalize_parser_action_name(raw: Any) -> str:
    """Normalize parser action names onto canonical router processor keys."""
    token = str(raw or "").strip().lower()
    return _ACTION_ALIASES.get(token, "")


def _canonicalize_actions(values: Sequence[Any]) -> Tuple[str, ...]:
    out: List[str] = []
    for value in values:
        action = normalize_parser_action_name(value)
        if not action or action in out:
            continue
        out.append(action)
    return tuple(out)


def _parse_actions_from_hint(hint: Mapping[str, Any]) -> Tuple[str, ...]:
    recommended = hint.get("recommended_processors")
    if isinstance(recommended, Sequence) and not isinstance(recommended, (str, bytes, bytearray)):
        parsed = _canonicalize_actions(recommended)
        if parsed:
            return parsed

    route_name = str(hint.get("action") or "").strip().lower()
    from_route = _ACTION_BY_ROUTE_NAME.get(route_name)
    if from_route:
        return from_route
    return ()


def _get_metadata_dict(sample: Any) -> Dict[str, Any]:
    if isinstance(sample, dict):
        metadata = sample.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            sample["metadata"] = metadata
        return metadata

    metadata = getattr(sample, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        try:
            setattr(sample, "metadata", metadata)
        except Exception:
            pass
    return metadata


def _get_doc_id(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("doc_id") or sample.get("id") or "")
    return str(getattr(sample, "doc_id", "") or getattr(sample, "id", "") or "")


def _get_pages(sample: Any) -> Optional[List[str]]:
    pages = sample.get("pages") if isinstance(sample, dict) else getattr(sample, "pages", None)
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes, bytearray)):
        return None
    return [str(page or "") for page in pages]


def _get_page_assets(sample: Any) -> Optional[List[Dict[str, Any]]]:
    metadata = _get_metadata_dict(sample)
    raw_assets = metadata.get("page_assets")
    if not isinstance(raw_assets, Sequence) or isinstance(raw_assets, (str, bytes, bytearray)):
        return None
    assets: List[Dict[str, Any]] = []
    for entry in raw_assets:
        if isinstance(entry, Mapping):
            assets.append(dict(entry))
    return assets


def _set_pages(sample: Any, pages: List[str]) -> None:
    if isinstance(sample, dict):
        sample["pages"] = list(pages)
        return
    try:
        setattr(sample, "pages", list(pages))
    except Exception:
        return


def _set_text(sample: Any, text: str) -> None:
    if isinstance(sample, dict):
        sample["text"] = text
        return
    try:
        setattr(sample, "text", text)
    except Exception:
        return


def _rebuild_page_char_ranges(
    pages: Sequence[str],
    *,
    page_joiner: str,
) -> Tuple[str, List[List[int]]]:
    joiner = str(page_joiner or "")
    ranges: List[List[int]] = []
    cursor = 0
    normalized_pages = [str(page or "") for page in pages]
    for idx, page in enumerate(normalized_pages):
        start = cursor
        cursor += len(page)
        end = cursor
        ranges.append([start, end])
        if idx + 1 < len(normalized_pages):
            cursor += len(joiner)
    return joiner.join(normalized_pages), ranges


def _post_json(url: str, payload: Dict[str, Any], *, timeout_seconds: float) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(request, timeout=float(timeout_seconds)) as response:
        data = response.read().decode("utf-8")
    if not data:
        return {}
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        return {"raw_response": data}
    if isinstance(parsed, dict):
        return parsed
    return {"response": parsed}


def _is_numeric_sequence(values: Any) -> bool:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return False
    for value in values:
        if isinstance(value, bool):
            return False
        if not isinstance(value, (int, float)):
            return False
    return True


def _validate_response_contract(
    *,
    action: str,
    response: Mapping[str, Any],
    strict_contracts: bool,
    contract_version: int,
) -> Tuple[bool, str]:
    status = str(response.get("status") or "").strip().lower()

    if strict_contracts:
        if int(response.get("contract_version") or -1) != int(contract_version):
            return False, "response contract_version mismatch"
        if str(response.get("response_type") or "").strip() != PARSER_ROUTER_RESPONSE_TYPE:
            return False, "response_type missing or invalid"
        if not status:
            return False, "response status missing"
        response_action = str(response.get("action") or "").strip().lower()
        if response_action and response_action != action:
            return False, "response action mismatch"

    if status and status in {"error", "skipped"}:
        return True, "ok"

    if action in {"ocr", "vlm_parse"}:
        has_page_texts = isinstance(response.get("page_texts"), Sequence) and not isinstance(
            response.get("page_texts"),
            (str, bytes, bytearray),
        )
        has_page_text = isinstance(response.get("page_text"), str)
        has_text = isinstance(response.get("text"), str)
        if strict_contracts and not (has_page_texts or has_page_text or has_text):
            return False, "missing textual extraction payload"

    if action == "vision_embedding":
        embedding = response.get("embedding")
        embeddings = response.get("embeddings")
        has_embedding = _is_numeric_sequence(embedding)
        has_embeddings = (
            isinstance(embeddings, Sequence)
            and not isinstance(embeddings, (str, bytes, bytearray))
            and all(_is_numeric_sequence(item) for item in embeddings)
        )
        if strict_contracts and not (has_embedding or has_embeddings):
            return False, "missing embedding payload"

    return True, "ok"


@dataclass
class ParserRouterConfig:
    """Configuration for parser-action routing."""

    enabled: bool = False
    fail_open: bool = True
    max_hints_per_sample: int = 128
    store_max_results_per_sample: int = 200
    enabled_processors: Tuple[str, ...] = DEFAULT_ROUTER_ACTIONS
    ocr_endpoint: Optional[str] = None
    vlm_endpoint: Optional[str] = None
    vision_embedding_endpoint: Optional[str] = None
    vlm_segment_endpoint: Optional[str] = None
    timeout_seconds: float = 20.0
    max_concurrency: int = 4
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    strict_contracts: bool = True
    contract_version: int = PARSER_ROUTER_CONTRACT_VERSION


@dataclass
class ParserActionResult:
    """Result payload from one routed parser action."""

    action: str
    processor: str
    status: str
    hint_index: int
    hint_source: str
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParserActionProcessor:
    """Base class for parser action processors."""

    action: str = ""

    def process(
        self,
        *,
        sample: Any,
        hint: Mapping[str, Any],
        hint_index: int,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        strict_contracts: bool,
        contract_version: int,
    ) -> ParserActionResult:
        raise NotImplementedError


class ExternalJSONHintProcessor(ParserActionProcessor):
    """
    Processor that forwards a hint payload to an external JSON endpoint.

    Endpoint contract is versioned; strict validation is configurable.
    """

    def __init__(self, action: str, endpoint: Optional[str]):
        self.action = str(action)
        self.endpoint = str(endpoint or "").strip() or None

    def _payload(
        self,
        *,
        sample: Any,
        hint: Mapping[str, Any],
        hint_index: int,
        contract_version: int,
    ) -> Dict[str, Any]:
        doc_id = _get_doc_id(sample)
        metadata = _get_metadata_dict(sample)
        pages = _get_pages(sample)
        page_assets = _get_page_assets(sample)

        start = int(hint.get("start", 0) or 0)
        end = int(hint.get("end", start + 1) or (start + 1))
        lo = 0
        hi = 0
        page_slice: List[str] = []
        asset_slice: List[Dict[str, Any]] = []
        if pages is not None and end > start:
            lo = max(0, min(len(pages), start))
            hi = max(lo, min(len(pages), end))
            page_slice = list(pages[lo:hi])
        if page_assets is not None and end > start:
            lo_assets = max(0, min(len(page_assets), start))
            hi_assets = max(lo_assets, min(len(page_assets), end))
            asset_slice = [dict(asset) for asset in page_assets[lo_assets:hi_assets]]

        return {
            "contract_version": int(contract_version),
            "request_type": PARSER_ROUTER_REQUEST_TYPE,
            "action": self.action,
            "doc_id": doc_id,
            "hint_index": int(hint_index),
            "hint": dict(hint),
            "sample": {
                "modality": (sample.get("modality") if isinstance(sample, dict) else getattr(sample, "modality", "text")),
                "source_path": metadata.get("source_path"),
                "parser_backend": metadata.get("parser_backend"),
                "page_range": {"start": lo, "end": hi},
                "pages": page_slice,
                "page_assets": asset_slice,
                "text": (sample.get("text") if isinstance(sample, dict) else getattr(sample, "text", "")),
            },
        }

    def _maybe_apply_text_update(
        self,
        *,
        sample: Any,
        hint: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> Dict[str, Any]:
        pages = _get_pages(sample)
        if pages is None:
            return {}

        axis_unit = str(hint.get("axis_unit") or hint.get("unit") or "").strip().lower()
        if axis_unit != "page":
            return {}

        start = int(hint.get("start", 0) or 0)
        end = int(hint.get("end", start + 1) or (start + 1))
        if end <= start:
            return {}

        lo = max(0, min(len(pages), start))
        hi = max(lo, min(len(pages), end))
        width = hi - lo
        if width <= 0:
            return {}

        replacement_pages: Optional[List[str]] = None
        raw_page_texts = response.get("page_texts")
        if isinstance(raw_page_texts, Sequence) and not isinstance(raw_page_texts, (str, bytes, bytearray)):
            values = [str(value or "") for value in raw_page_texts]
            if len(values) == width:
                replacement_pages = values

        if replacement_pages is None and width == 1:
            single_text = response.get("page_text")
            if single_text is None:
                single_text = response.get("text")
            if isinstance(single_text, str):
                replacement_pages = [single_text]

        if replacement_pages is None:
            return {}

        pages[lo:hi] = replacement_pages
        _set_pages(sample, pages)
        metadata = _get_metadata_dict(sample)
        joiner = str(metadata.get("page_joiner") or "\n\n")
        rebuilt_text, rebuilt_ranges = _rebuild_page_char_ranges(pages, page_joiner=joiner)
        _set_text(sample, rebuilt_text)
        metadata["page_char_ranges"] = rebuilt_ranges
        axis_ranges = metadata.get("axis_char_ranges")
        if not isinstance(axis_ranges, dict):
            axis_ranges = {}
            metadata["axis_char_ranges"] = axis_ranges
        axis_ranges["page"] = rebuilt_ranges
        metadata["page_count"] = len(pages)
        metadata["page_joiner"] = joiner
        return {
            "applied_text_update": True,
            "updated_page_start": lo,
            "updated_page_end": hi,
            "updated_page_count": len(replacement_pages),
        }

    def process(
        self,
        *,
        sample: Any,
        hint: Mapping[str, Any],
        hint_index: int,
        timeout_seconds: float,
        max_retries: int,
        retry_backoff_seconds: float,
        strict_contracts: bool,
        contract_version: int,
    ) -> ParserActionResult:
        hint_source = str(hint.get("source") or "parser_feedback_hint")
        if not self.endpoint:
            return ParserActionResult(
                action=self.action,
                processor=self.action,
                status="skipped_unconfigured",
                hint_index=hint_index,
                hint_source=hint_source,
                message=f"No endpoint configured for processor '{self.action}'",
                metadata={},
            )

        payload = self._payload(
            sample=sample,
            hint=hint,
            hint_index=hint_index,
            contract_version=contract_version,
        )

        response: Dict[str, Any] = {}
        error_message: Optional[str] = None
        attempt_count = 0
        total_latency_ms = 0.0
        for attempt in range(max(0, int(max_retries)) + 1):
            attempt_count = attempt + 1
            t0 = time.perf_counter()
            try:
                response = _post_json(self.endpoint, payload, timeout_seconds=timeout_seconds)
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                error_message = None
                break
            except urllib_error.URLError as exc:
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                error_message = f"Endpoint request failed: {exc}"
            except Exception as exc:
                total_latency_ms += (time.perf_counter() - t0) * 1000.0
                error_message = f"Processor error: {exc}"

            if attempt < max(0, int(max_retries)):
                sleep_seconds = max(0.0, float(retry_backoff_seconds)) * (2.0 ** attempt)
                time.sleep(sleep_seconds)

        if error_message is not None:
            return ParserActionResult(
                action=self.action,
                processor=self.action,
                status="error",
                hint_index=hint_index,
                hint_source=hint_source,
                message=error_message,
                metadata={
                    "endpoint": self.endpoint,
                    "attempt_count": attempt_count,
                    "latency_ms": round(total_latency_ms, 3),
                },
            )

        if not isinstance(response, dict):
            return ParserActionResult(
                action=self.action,
                processor=self.action,
                status="invalid_response_contract",
                hint_index=hint_index,
                hint_source=hint_source,
                message="Endpoint returned non-dict payload",
                metadata={
                    "endpoint": self.endpoint,
                    "attempt_count": attempt_count,
                    "latency_ms": round(total_latency_ms, 3),
                },
            )

        valid_contract, contract_message = _validate_response_contract(
            action=self.action,
            response=response,
            strict_contracts=bool(strict_contracts),
            contract_version=int(contract_version),
        )
        if not valid_contract:
            return ParserActionResult(
                action=self.action,
                processor=self.action,
                status="invalid_response_contract",
                hint_index=hint_index,
                hint_source=hint_source,
                message=contract_message,
                metadata={
                    "endpoint": self.endpoint,
                    "attempt_count": attempt_count,
                    "latency_ms": round(total_latency_ms, 3),
                    "response_keys": sorted(str(key) for key in response.keys())[:32],
                },
            )

        response_status = str(response.get("status") or "ok").strip().lower()
        normalized_status = "applied"
        if response_status in {"error", "skipped"}:
            normalized_status = response_status

        result_metadata: Dict[str, Any] = {
            "endpoint": self.endpoint,
            "attempt_count": attempt_count,
            "latency_ms": round(total_latency_ms, 3),
            "response_contract_version": response.get("contract_version"),
            "response_status": response_status,
            "response_keys": sorted(str(key) for key in response.keys())[:32],
        }
        if self.action in {"ocr", "vlm_parse"}:
            with _SAMPLE_TEXT_UPDATE_LOCK:
                result_metadata.update(
                    self._maybe_apply_text_update(sample=sample, hint=hint, response=response)
                )
        else:
            result_metadata.update(self._maybe_apply_text_update(sample=sample, hint=hint, response=response))

        if self.action == "vision_embedding":
            embedding = response.get("embedding")
            if _is_numeric_sequence(embedding):
                result_metadata["embedding_dim"] = len(embedding)
            embeddings = response.get("embeddings")
            if isinstance(embeddings, Sequence) and not isinstance(embeddings, (str, bytes, bytearray)):
                result_metadata["embeddings_count"] = len(embeddings)

        message = str(response.get("message") or "Action dispatched")
        return ParserActionResult(
            action=self.action,
            processor=self.action,
            status=normalized_status,
            hint_index=hint_index,
            hint_source=hint_source,
            message=message,
            metadata=result_metadata,
        )


class ParserRouter:
    """Routes parser feedback hints to action processors."""

    def __init__(
        self,
        config: ParserRouterConfig,
        *,
        processors: Optional[Mapping[str, ParserActionProcessor]] = None,
    ):
        self.config = config
        if processors is None:
            processors = {
                "ocr": ExternalJSONHintProcessor("ocr", endpoint=config.ocr_endpoint),
                "vlm_parse": ExternalJSONHintProcessor("vlm_parse", endpoint=config.vlm_endpoint),
                "vision_embedding": ExternalJSONHintProcessor(
                    "vision_embedding",
                    endpoint=config.vision_embedding_endpoint,
                ),
            }
        self.processors = dict(processors)

    def route_sample(self, sample: Any) -> Dict[str, Any]:
        metadata = _get_metadata_dict(sample)
        parser_feedback = metadata.get("parser_feedback")
        raw_hints = parser_feedback.get("axis_hints") if isinstance(parser_feedback, dict) else None
        hints: List[Dict[str, Any]] = []
        if isinstance(raw_hints, Sequence) and not isinstance(raw_hints, (str, bytes, bytearray)):
            for hint in raw_hints:
                if isinstance(hint, dict):
                    hints.append(hint)

        max_hints = max(0, int(self.config.max_hints_per_sample))
        hints = hints[:max_hints] if max_hints > 0 else []

        results: List[Tuple[int, ParserActionResult]] = []
        hint_count_with_actions = 0
        jobs: List[Tuple[int, int, Dict[str, Any], str, ParserActionProcessor]] = []
        job_index = 0

        for hint_index, hint in enumerate(hints):
            actions = _parse_actions_from_hint(hint)
            if not actions:
                continue
            hint_count_with_actions += 1
            for action in actions:
                if action not in self.config.enabled_processors:
                    results.append(
                        (
                            job_index,
                            ParserActionResult(
                                action=action,
                                processor=action,
                                status="skipped_action_filtered",
                                hint_index=hint_index,
                                hint_source=str(hint.get("source") or "parser_feedback_hint"),
                                message=f"Action '{action}' not enabled in parser router config",
                                metadata={},
                            ),
                        )
                    )
                    job_index += 1
                    continue

                processor = self.processors.get(action)
                if processor is None:
                    results.append(
                        (
                            job_index,
                            ParserActionResult(
                                action=action,
                                processor=action,
                                status="skipped_missing_processor",
                                hint_index=hint_index,
                                hint_source=str(hint.get("source") or "parser_feedback_hint"),
                                message=f"No processor registered for action '{action}'",
                                metadata={},
                            ),
                        )
                    )
                    job_index += 1
                    continue

                jobs.append((job_index, hint_index, dict(hint), action, processor))
                job_index += 1

        def _execute_job(job: Tuple[int, int, Dict[str, Any], str, ParserActionProcessor]) -> Tuple[int, ParserActionResult]:
            local_job_index, hint_index, hint, action, processor = job
            try:
                result = processor.process(
                    sample=sample,
                    hint=hint,
                    hint_index=hint_index,
                    timeout_seconds=self.config.timeout_seconds,
                    max_retries=self.config.max_retries,
                    retry_backoff_seconds=self.config.retry_backoff_seconds,
                    strict_contracts=self.config.strict_contracts,
                    contract_version=self.config.contract_version,
                )
            except Exception as exc:
                if self.config.fail_open:
                    logger.warning("Parser router processor '%s' failed: %s", action, exc)
                    result = ParserActionResult(
                        action=action,
                        processor=action,
                        status="error",
                        hint_index=hint_index,
                        hint_source=str(hint.get("source") or "parser_feedback_hint"),
                        message=str(exc),
                        metadata={},
                    )
                else:
                    raise
            return local_job_index, result

        max_workers = max(1, int(self.config.max_concurrency))
        if jobs and max_workers > 1 and len(jobs) > 1:
            with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as pool:
                future_map = {pool.submit(_execute_job, job): job[0] for job in jobs}
                for future in as_completed(future_map):
                    results.append(future.result())
        else:
            for job in jobs:
                results.append(_execute_job(job))

        results.sort(key=lambda item: item[0])
        ordered_results = [result for _, result in results]

        status_counts: Dict[str, int] = {}
        latencies_ms: List[float] = []
        for result in ordered_results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            latency_value = result.metadata.get("latency_ms") if isinstance(result.metadata, dict) else None
            if isinstance(latency_value, (int, float)):
                latencies_ms.append(float(latency_value))

        applied = status_counts.get("applied", 0)
        errors = status_counts.get("error", 0) + status_counts.get("invalid_response_contract", 0)
        skipped = sum(
            count
            for key, count in status_counts.items()
            if key.startswith("skipped")
        )

        summary = {
            "hint_count": len(hints),
            "hint_count_with_actions": hint_count_with_actions,
            "actions_attempted": len(ordered_results),
            "applied": applied,
            "skipped": skipped,
            "errors": errors,
            "status_counts": status_counts,
            "latency_ms_mean": (sum(latencies_ms) / len(latencies_ms)) if latencies_ms else None,
            "latency_ms_p95": (sorted(latencies_ms)[int(max(0, min(len(latencies_ms) - 1, round(0.95 * (len(latencies_ms) - 1)))))] if latencies_ms else None),
        }

        store_limit = max(0, int(self.config.store_max_results_per_sample))
        stored_results = [asdict(result) for result in ordered_results[:store_limit]]
        dropped_results = max(0, len(ordered_results) - len(stored_results))

        run_record = {
            "ran_at": datetime.now(timezone.utc).isoformat(),
            "doc_id": _get_doc_id(sample),
            "summary": summary,
            "config": {
                "enabled_processors": list(self.config.enabled_processors),
                "timeout_seconds": self.config.timeout_seconds,
                "max_hints_per_sample": self.config.max_hints_per_sample,
                "max_concurrency": self.config.max_concurrency,
                "max_retries": self.config.max_retries,
                "retry_backoff_seconds": self.config.retry_backoff_seconds,
                "strict_contracts": self.config.strict_contracts,
                "contract_version": self.config.contract_version,
            },
            "results": stored_results,
            "results_truncated": dropped_results,
        }

        router_state = metadata.get("parser_router")
        if not isinstance(router_state, dict):
            router_state = {}
            metadata["parser_router"] = router_state
        previous_runs = int(router_state.get("run_count") or 0)
        router_state["run_count"] = previous_runs + 1
        router_state["last_run"] = run_record

        if isinstance(parser_feedback, MutableMapping):
            parser_feedback["router"] = {
                "run_count": router_state["run_count"],
                "last_run_summary": summary,
                "enabled_processors": list(self.config.enabled_processors),
                "contract_version": self.config.contract_version,
            }

        return summary

    def route_samples(self, samples: Sequence[Any]) -> Dict[str, Any]:
        aggregate = {
            "enabled": bool(self.config.enabled),
            "docs_total": int(len(samples)),
            "docs_with_hints": 0,
            "docs_routed": 0,
            "hints_seen": 0,
            "hints_with_actions": 0,
            "actions_attempted": 0,
            "applied": 0,
            "skipped": 0,
            "errors": 0,
        }

        for sample in samples:
            metadata = _get_metadata_dict(sample)
            parser_feedback = metadata.get("parser_feedback")
            raw_hints = parser_feedback.get("axis_hints") if isinstance(parser_feedback, dict) else None
            hint_count = (
                len(raw_hints)
                if isinstance(raw_hints, Sequence) and not isinstance(raw_hints, (str, bytes, bytearray))
                else 0
            )
            if hint_count > 0:
                aggregate["docs_with_hints"] += 1

            summary = self.route_sample(sample)
            aggregate["docs_routed"] += 1
            aggregate["hints_seen"] += int(summary.get("hint_count") or 0)
            aggregate["hints_with_actions"] += int(summary.get("hint_count_with_actions") or 0)
            aggregate["actions_attempted"] += int(summary.get("actions_attempted") or 0)
            aggregate["applied"] += int(summary.get("applied") or 0)
            aggregate["skipped"] += int(summary.get("skipped") or 0)
            aggregate["errors"] += int(summary.get("errors") or 0)

        return aggregate
