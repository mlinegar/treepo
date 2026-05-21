from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence
from urllib.parse import urlparse

from treepo._research.core.engines import EngineSurface, EngineType, default_engine_port
from treepo._research.core.inference_engine import InferenceEngine, build_inference_engine
from treepo._research.runtime.calls import RuntimeCallScheduler
from treepo._research.runtime.contracts import (
    ChatInput,
    InferenceResponse,
    ModelResponse,
    RUNTIME_ROLE_EMBEDDER,
    RUNTIME_ROLE_SCORER,
    RUNTIME_ROLE_STATE_MODEL,
    RUNTIME_ROLE_SUMMARIZER,
    RuntimeSurfaceCall,
)


_SURFACE_ALIASES = {
    "chat": EngineSurface.CHAT_OPENAI.value,
    "chat_openai": EngineSurface.CHAT_OPENAI.value,
    "llm": EngineSurface.CHAT_OPENAI.value,
    "embedder": EngineSurface.EMBEDDING.value,
    "embedding": EngineSurface.EMBEDDING.value,
    "embeddings": EngineSurface.EMBEDDING.value,
    "embed": EngineSurface.EMBEDDING.value,
    "state": EngineSurface.OPERATOR.value,
    "state_model": EngineSurface.OPERATOR.value,
    "operator": EngineSurface.OPERATOR.value,
    "operators": EngineSurface.OPERATOR.value,
}

_ROLE_TO_SURFACE = {
    RUNTIME_ROLE_SCORER: EngineSurface.CHAT_OPENAI.value,
    RUNTIME_ROLE_SUMMARIZER: EngineSurface.CHAT_OPENAI.value,
    RUNTIME_ROLE_EMBEDDER: EngineSurface.EMBEDDING.value,
    RUNTIME_ROLE_STATE_MODEL: EngineSurface.OPERATOR.value,
}

_PUBLIC_RUNTIME_KEYS = {
    "experiment_id",
    "experiment",
    "benchmark",
    "methods",
    "scorer",
    "summarizer",
    "embedder",
    "state_model",
    "oracle",
    "runtime_defaults",
    "phases",
}

_PUBLIC_PHASE_KEYS = {
    "phase_id",
    "tasks",
    "lengths",
    "seeds",
    "num_samples",
    "split",
    "methods",
    "runtime_overrides",
    "benchmark_overrides",
    "runtime_grid",
    "benchmark_grid",
}

_PUBLIC_RUNTIME_KEY_HINT = (
    "Supported runtime-eval config keys are: "
    + ", ".join(sorted(_PUBLIC_RUNTIME_KEYS))
    + "."
)


def _surface_key(surface: EngineSurface | str) -> str:
    if isinstance(surface, EngineSurface):
        return surface.value
    raw = str(surface or "").strip().lower().replace("-", "_")
    return _SURFACE_ALIASES.get(raw, raw)


def _infer_chat_engine(base_url: str, explicit: Any = None) -> str:
    if explicit:
        return str(explicit)
    parsed = urlparse(str(base_url or ""))
    host = str(parsed.netloc or "").lower()
    if "api.openai.com" in host:
        return EngineType.OPENAI.value
    if parsed.port is not None:
        try:
            if parsed.port == default_engine_port(EngineType.VLLM, role="task"):
                return EngineType.VLLM.value
            if parsed.port == default_engine_port(EngineType.SGLANG, role="task"):
                return EngineType.SGLANG.value
        except Exception:
            pass
    if base_url:
        return EngineType.CUSTOM_HTTP.value
    return EngineType.VLLM.value


def _endpoint(cfg: Mapping[str, Any]) -> str:
    return str(cfg.get("base_url") or cfg.get("endpoint") or cfg.get("api_base") or "")


def _infer_operator_engine(base_url: str, explicit: Any = None) -> str:
    if explicit:
        rendered = str(explicit).strip().lower().replace("-", "_")
        if rendered in {"native", "neural_operator", "state_model", "pytorch", "torch"}:
            return EngineType.NATIVE_OPERATOR.value
        if rendered in {"served", "http", "custom", "custom_http"}:
            return EngineType.CUSTOM_HTTP.value
        return str(explicit)
    return EngineType.CUSTOM_HTTP.value if base_url else EngineType.NATIVE_OPERATOR.value


def validate_public_runtime_config(spec: Mapping[str, Any]) -> None:
    """Validate the public runtime-eval v2 config shape."""

    for key in sorted(str(item) for item in spec.keys()):
        if key not in _PUBLIC_RUNTIME_KEYS:
            raise ValueError(
                f"Unsupported runtime-eval config key: {key}. "
                f"{_PUBLIC_RUNTIME_KEY_HINT}"
            )
    for idx, phase in enumerate(list(spec.get("phases", []) or [])):
        if not isinstance(phase, Mapping):
            continue
        phase_id = str(phase.get("phase_id", idx))
        for key in sorted(str(item) for item in phase.keys()):
            if key not in _PUBLIC_PHASE_KEYS:
                supported = ", ".join(sorted(_PUBLIC_PHASE_KEYS))
                raise ValueError(
                    f"Unsupported runtime-eval phase key in {phase_id!r}: {key}. "
                    f"Supported phase keys are: {supported}."
                )


def _normalize_chat_role(raw_cfg: Mapping[str, Any], *, role: str) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    base_url = _endpoint(cfg)
    out = {
        "role": role,
        "surface": EngineSurface.CHAT_OPENAI.value,
        "engine": _infer_chat_engine(base_url, cfg.get("engine") or cfg.get("provider")),
        "base_url": base_url or "http://localhost:8000/v1",
        "model": str(cfg.get("model", "default") or "default"),
        "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
        "temperature": float(cfg.get("temperature", 0.0) or 0.0),
        "timeout": float(cfg.get("timeout", 120.0) or 120.0),
        "enable_cache": bool(cfg.get("enable_cache", True)),
    }
    if cfg.get("batch_size") is not None:
        out["batch_size"] = int(cfg.get("batch_size") or 0)
    for key in ("max_context", "context_window", "max_model_len", "supports_lora"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def _normalize_embedder_role(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    base_url = _endpoint(cfg)
    return {
        "role": RUNTIME_ROLE_EMBEDDER,
        "surface": EngineSurface.EMBEDDING.value,
        "engine": _infer_chat_engine(
            base_url, cfg.get("engine") or cfg.get("provider") or "vllm"
        ),
        "base_url": base_url or "http://localhost:8003/v1",
        "model": str(cfg.get("model", "default") or "default"),
        "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
        "timeout": float(cfg.get("timeout", 60.0) or 60.0),
        "mock": bool(cfg.get("mock", False)),
        "mock_dim": int(cfg.get("mock_dim", 32) or 32),
        "batch_size": int(cfg.get("batch_size", 32) or 32),
    }


def _normalize_state_model_role(raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    cfg = dict(raw_cfg)
    base_url = _endpoint(cfg)
    operator_kind = str(cfg.get("kind", "") or "")
    explicit_engine = cfg.get("engine") or cfg.get("provider")
    if explicit_engine is None and not base_url:
        explicit_engine = operator_kind
    return {
        "role": RUNTIME_ROLE_STATE_MODEL,
        "surface": EngineSurface.OPERATOR.value,
        "engine": _infer_operator_engine(base_url, explicit_engine),
        "model": str(cfg.get("model") or cfg.get("name") or operator_kind or "state_model"),
        "base_url": base_url,
        "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
        "timeout": float(cfg.get("timeout", 120.0) or 120.0),
        "checkpoint_path": str(cfg.get("checkpoint_path") or cfg.get("checkpoint") or ""),
        "execution_mode": str(cfg.get("execution_mode", "auto") or "auto"),
        "kind": operator_kind,
        "device": str(cfg.get("device", "auto") or "auto"),
    }


def _surface_from_role_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in dict(cfg).items() if key not in {"role", "surface"}}


def _normalize_internal_surface(raw_key: str, raw_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    key = _surface_key(raw_key)
    cfg = dict(raw_cfg)
    if key == EngineSurface.CHAT_OPENAI.value:
        out = dict(cfg)
        out.update({
            "engine": _infer_chat_engine(
                str(cfg.get("base_url", "") or ""), cfg.get("engine") or cfg.get("provider")
            ),
            "base_url": str(cfg.get("base_url", "") or ""),
            "model": str(cfg.get("model", "default") or "default"),
            "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
            "temperature": float(cfg.get("temperature", 0.0) or 0.0),
            "timeout": float(cfg.get("timeout", 120.0) or 120.0),
            "enable_cache": bool(cfg.get("enable_cache", True)),
        })
        return out
    if key == EngineSurface.EMBEDDING.value:
        out = dict(cfg)
        out.update({
            "engine": _infer_chat_engine(
                str(cfg.get("base_url", "") or ""), cfg.get("engine") or "vllm"
            ),
            "base_url": str(cfg.get("base_url", "") or ""),
            "model": str(cfg.get("model", "default") or "default"),
            "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
            "timeout": float(cfg.get("timeout", 60.0) or 60.0),
            "mock": bool(cfg.get("mock", False)),
            "mock_dim": int(cfg.get("mock_dim", 32) or 32),
            "batch_size": int(cfg.get("batch_size", 32) or 32),
        })
        return out
    if key == EngineSurface.OPERATOR.value:
        out = dict(cfg)
        out.update({
            "engine": str(cfg.get("engine", "native_operator") or "native_operator"),
            "model": str(cfg.get("model", "operator") or "operator"),
            "base_url": str(cfg.get("base_url", "") or ""),
            "api_key": str(cfg.get("api_key", "EMPTY") or "EMPTY"),
            "timeout": float(cfg.get("timeout", 120.0) or 120.0),
            "checkpoint_path": str(cfg.get("checkpoint_path", "") or ""),
            "execution_mode": str(cfg.get("execution_mode", "auto") or "auto"),
        })
        return out
    return cfg


def normalize_role_config(
    spec: Mapping[str, Any], *, allow_internal_surfaces: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Normalize paper-facing runtime roles."""

    if not allow_internal_surfaces:
        validate_public_runtime_config(spec)

    roles: Dict[str, Dict[str, Any]] = {}
    raw_roles = dict(spec.get("roles", {}) or {}) if allow_internal_surfaces else {}
    for role, raw_cfg in raw_roles.items():
        if isinstance(raw_cfg, Mapping):
            cfg = dict(raw_cfg)
            cfg.setdefault("role", str(role))
            cfg.setdefault("surface", _ROLE_TO_SURFACE.get(str(role), ""))
            roles[str(role)] = cfg

    scorer_cfg = dict(spec.get(RUNTIME_ROLE_SCORER, {}) or {})
    if scorer_cfg:
        roles[RUNTIME_ROLE_SCORER] = _normalize_chat_role(
            scorer_cfg, role=RUNTIME_ROLE_SCORER
        )

    summarizer_cfg = dict(spec.get(RUNTIME_ROLE_SUMMARIZER, {}) or {})
    if summarizer_cfg:
        roles[RUNTIME_ROLE_SUMMARIZER] = _normalize_chat_role(
            summarizer_cfg, role=RUNTIME_ROLE_SUMMARIZER
        )
    elif RUNTIME_ROLE_SCORER in roles and RUNTIME_ROLE_SUMMARIZER not in roles:
        defaulted = dict(roles[RUNTIME_ROLE_SCORER])
        defaulted["role"] = RUNTIME_ROLE_SUMMARIZER
        defaulted["defaulted_from"] = RUNTIME_ROLE_SCORER
        roles[RUNTIME_ROLE_SUMMARIZER] = defaulted

    embedder_cfg = dict(spec.get(RUNTIME_ROLE_EMBEDDER, {}) or {})
    if embedder_cfg:
        roles[RUNTIME_ROLE_EMBEDDER] = _normalize_embedder_role(embedder_cfg)

    state_model_cfg = dict(spec.get(RUNTIME_ROLE_STATE_MODEL, {}) or {})
    if state_model_cfg:
        roles[RUNTIME_ROLE_STATE_MODEL] = _normalize_state_model_role(state_model_cfg)

    if allow_internal_surfaces:
        raw_surfaces = dict(spec.get("surfaces", {}) or {})
        if raw_surfaces:
            normalized_surfaces = {
                _surface_key(str(key)): _normalize_internal_surface(str(key), value)
                for key, value in raw_surfaces.items()
                if isinstance(value, Mapping)
            }
            if RUNTIME_ROLE_SCORER not in roles and EngineSurface.CHAT_OPENAI.value in normalized_surfaces:
                cfg = dict(normalized_surfaces[EngineSurface.CHAT_OPENAI.value])
                cfg.update({"role": RUNTIME_ROLE_SCORER, "surface": EngineSurface.CHAT_OPENAI.value})
                roles[RUNTIME_ROLE_SCORER] = cfg
            if (
                RUNTIME_ROLE_SUMMARIZER not in roles
                and RUNTIME_ROLE_SCORER in roles
            ):
                cfg = dict(roles[RUNTIME_ROLE_SCORER])
                cfg["role"] = RUNTIME_ROLE_SUMMARIZER
                cfg["defaulted_from"] = RUNTIME_ROLE_SCORER
                roles[RUNTIME_ROLE_SUMMARIZER] = cfg
            if RUNTIME_ROLE_EMBEDDER not in roles and EngineSurface.EMBEDDING.value in normalized_surfaces:
                cfg = dict(normalized_surfaces[EngineSurface.EMBEDDING.value])
                cfg.update({"role": RUNTIME_ROLE_EMBEDDER, "surface": EngineSurface.EMBEDDING.value})
                roles[RUNTIME_ROLE_EMBEDDER] = cfg
            if RUNTIME_ROLE_STATE_MODEL not in roles and EngineSurface.OPERATOR.value in normalized_surfaces:
                cfg = dict(normalized_surfaces[EngineSurface.OPERATOR.value])
                cfg.update({"role": RUNTIME_ROLE_STATE_MODEL, "surface": EngineSurface.OPERATOR.value})
                roles[RUNTIME_ROLE_STATE_MODEL] = cfg

    return roles


def normalize_surface_config(
    spec: Mapping[str, Any], *, allow_internal_surfaces: bool = False
) -> Dict[str, Any]:
    """Normalize runtime roles into internal inference surfaces."""

    roles = normalize_role_config(spec, allow_internal_surfaces=allow_internal_surfaces)
    normalized: Dict[str, Any] = {}
    for role in (
        RUNTIME_ROLE_SCORER,
        RUNTIME_ROLE_EMBEDDER,
        RUNTIME_ROLE_STATE_MODEL,
        RUNTIME_ROLE_SUMMARIZER,
    ):
        cfg = roles.get(role)
        if not cfg:
            continue
        surface = str(cfg.get("surface") or _ROLE_TO_SURFACE.get(role, ""))
        if surface and surface not in normalized:
            normalized[surface] = _surface_from_role_cfg(cfg)

    return normalized


def normalize_oracle_config(spec: Mapping[str, Any]) -> Dict[str, Any]:
    oracle = dict(spec.get("oracle", {}) or {})
    if not oracle:
        benchmark = dict(spec.get("benchmark", {}) or {})
        if str(benchmark.get("name", "") or "") == "longbench_v2":
            oracle = {"kind": "benchmark_labels"}
    return oracle


def normalize_runtime_inference_spec(
    spec: Mapping[str, Any], *, allow_internal_surfaces: bool = True
) -> Dict[str, Any]:
    """Return a copy of a runtime spec with normalized `surfaces` attached."""

    out = dict(spec)
    out["roles"] = normalize_role_config(spec, allow_internal_surfaces=allow_internal_surfaces)
    out["surfaces"] = normalize_surface_config(
        out, allow_internal_surfaces=allow_internal_surfaces
    )
    out["oracle"] = normalize_oracle_config(out)
    return out


@dataclass(frozen=True)
class SurfaceCapabilities:
    surface: str
    engine: str
    model: str = "default"
    execution_mode: str = "served"
    supports_logprobs: bool = False
    supports_batching: bool = False
    supports_lora: bool = False
    max_context: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SurfaceBackboneAdapter:
    """BackboneAdapter-compatible shim over a chat runtime role."""

    def __init__(self, context: "RuntimeInferenceContext", *, role: str = RUNTIME_ROLE_SCORER) -> None:
        self.context = context
        self.role = str(role or RUNTIME_ROLE_SCORER)
        caps = context.role_capabilities(self.role)
        self.supports_logprobs = bool(caps.supports_logprobs)

    def model_id(self) -> str:
        return str(self.context.role_capabilities(self.role).model)

    def generate(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: Optional[float] = None,
        stop: Optional[Sequence[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        cfg = self.context.role_config(self.role)
        temp = float(cfg.get("temperature", 0.0) if temperature is None else temperature)
        extra_payload = dict(extra or {})
        request_kind = str(extra_payload.pop("request_kind", "") or "")
        node_id = str(extra_payload.pop("node_id", "") or "")
        if not request_kind:
            request_kind = "answer_logprobs" if extra_payload.get("logprobs") else "chat_completion"
        result = self.context.scheduler.schedule(
            RuntimeSurfaceCall(
                surface=EngineSurface.CHAT_OPENAI,
                input=ChatInput(
                    messages=[dict(message) for message in messages],
                    max_tokens=int(max_tokens),
                    temperature=temp,
                    stop=list(stop or []),
                    extra=extra_payload,
                ),
                role=self.role,
                request_kind=request_kind,
                node_id=node_id,
            ),
        )
        return result.response.to_model_response()


class SurfaceEmbeddingClient:
    """EmbeddingClient-compatible shim over an EMBEDDING surface."""

    def __init__(self, context: "RuntimeInferenceContext") -> None:
        self.context = context

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self.context.scheduler.embed_texts(texts)


class RuntimeInferenceContext:
    """Surface router above InferenceEngine for runtime methods."""

    def __init__(
        self, spec: Mapping[str, Any], *, mock: bool = False, call_sink: Any = None
    ) -> None:
        self.spec = normalize_runtime_inference_spec(spec)
        self.mock = bool(mock)
        self._surface_cfg = dict(self.spec.get("surfaces", {}) or {})
        self._role_cfg = dict(self.spec.get("roles", {}) or {})
        self._oracle_cfg = dict(self.spec.get("oracle", {}) or {})
        self._engines: Dict[str, InferenceEngine] = {}
        self._backbones: Dict[str, SurfaceBackboneAdapter] = {}
        self._embedding_client: Optional[SurfaceEmbeddingClient] = None
        self._call_scope: Dict[str, Any] = {}
        self.scheduler = RuntimeCallScheduler(self, sink=call_sink)

    @property
    def surfaces(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._surface_cfg.items()}

    @property
    def roles(self) -> Dict[str, Dict[str, Any]]:
        return {key: dict(value) for key, value in self._role_cfg.items()}

    @property
    def oracle(self) -> Dict[str, Any]:
        return dict(self._oracle_cfg)

    def has_surface(self, surface: EngineSurface | str) -> bool:
        return _surface_key(surface) in self._surface_cfg

    def has_role(self, role: str) -> bool:
        return str(role or "") in self._role_cfg

    def surface_config(self, surface: EngineSurface | str) -> Dict[str, Any]:
        key = _surface_key(surface)
        if key not in self._surface_cfg:
            raise RuntimeError(f"No runtime inference surface configured for {key!r}.")
        return dict(self._surface_cfg[key])

    def role_config(self, role: str) -> Dict[str, Any]:
        key = str(role or "")
        if key not in self._role_cfg:
            raise RuntimeError(f"No runtime inference role configured for {key!r}.")
        return dict(self._role_cfg[key])

    def surface_for_role(self, role: str) -> str:
        cfg = self.role_config(role)
        return str(cfg.get("surface") or _ROLE_TO_SURFACE.get(str(role), ""))

    def call_scope(self) -> Dict[str, Any]:
        return dict(self._call_scope)

    def set_call_scope(self, **metadata: Any) -> Dict[str, Any]:
        previous = dict(self._call_scope)
        self._call_scope = {
            str(key): value for key, value in dict(metadata).items() if value is not None
        }
        return previous

    def capabilities(self, surface: EngineSurface | str) -> SurfaceCapabilities:
        key = _surface_key(surface)
        cfg = self.surface_config(key)
        return self._capabilities_from_config(key, cfg)

    def role_capabilities(self, role: str) -> SurfaceCapabilities:
        cfg = self.role_config(role)
        surface = str(cfg.get("surface") or _ROLE_TO_SURFACE.get(str(role), ""))
        return self._capabilities_from_config(surface, cfg)

    def _capabilities_from_config(
        self, surface: str, cfg: Mapping[str, Any]
    ) -> SurfaceCapabilities:
        key = _surface_key(surface)
        engine = str(cfg.get("engine", "") or "")
        model = str(cfg.get("model", "default") or "default")
        execution_mode = str(
            cfg.get("execution_mode") or ("native" if engine == "native_operator" else "served")
        )
        if key == EngineSurface.CHAT_OPENAI.value:
            supports_logprobs = (not self.mock) and engine in {
                "vllm",
                "sglang",
                "openai",
                "custom_http",
            }
            supports_batching = engine in {"vllm", "sglang", "custom_http"}
        else:
            supports_logprobs = False
            supports_batching = bool(cfg.get("batch_size")) or key == EngineSurface.EMBEDDING.value
        max_context = (
            cfg.get("max_context") or cfg.get("context_window") or cfg.get("max_model_len")
        )
        try:
            max_context_int = int(max_context) if max_context is not None else None
        except (TypeError, ValueError):
            max_context_int = None
        return SurfaceCapabilities(
            surface=key,
            engine=engine,
            model=model,
            execution_mode=execution_mode,
            supports_logprobs=supports_logprobs,
            supports_batching=supports_batching,
            supports_lora=bool(
                cfg.get("supports_lora") or cfg.get("lora") or cfg.get("adapter_path")
            ),
            max_context=max_context_int,
        )

    def engine(self, surface: EngineSurface | str, *, role: str | None = None) -> InferenceEngine:
        key = _surface_key(surface)
        cache_key = key
        if role:
            role_cfg = self.role_config(role)
            role_surface = str(role_cfg.get("surface") or _ROLE_TO_SURFACE.get(str(role), ""))
            if role_surface and _surface_key(role_surface) == key:
                cfg = role_cfg
                cache_key = f"{key}:{role}"
            else:
                cfg = self.surface_config(key)
        else:
            cfg = self.surface_config(key)
        if cache_key in self._engines:
            return self._engines[cache_key]
        surface_enum = EngineSurface(key)
        engine_name = str(
            cfg.get("engine")
            or ("native_operator" if surface_enum is EngineSurface.OPERATOR else "vllm")
        )
        backend = None
        if surface_enum is EngineSurface.OPERATOR:
            backend = cfg.get("operator_client") or cfg.get("backend")
        elif surface_enum is EngineSurface.EMBEDDING:
            backend = cfg.get("embedding_client") or cfg.get("backend")
        engine = build_inference_engine(
            engine_name,
            surface=surface_enum,
            model=str(cfg.get("model", "default") or "default"),
            base_url=str(cfg.get("base_url", "") or "") or None,
            api_key=str(cfg.get("api_key", "EMPTY") or "EMPTY"),
            timeout=float(cfg.get("timeout", 120.0) or 120.0),
            mock=bool(self.mock or cfg.get("mock", False)),
            enable_cache=bool(cfg.get("enable_cache", True)),
            backend=backend,
        )
        self._engines[cache_key] = engine
        return engine

    def execute(
        self,
        surface: EngineSurface | str,
        input_payload: Any,
        *,
        request_id: str = "",
        document_id: str = "",
        routing_key: str = "",
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        engine_options: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        surface_enum = EngineSurface(_surface_key(surface))
        result = self.scheduler.schedule(
            RuntimeSurfaceCall(
                surface=surface_enum,
                input=input_payload,
                request_id=request_id,
                document_id=document_id,
                routing_key=routing_key,
                priority=int(priority),
                metadata=dict(metadata or {}),
                engine_options=dict(engine_options or {}),
            )
        )
        return result.response

    def execute_operator(
        self,
        operation: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        batch: Optional[Sequence[Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResponse:
        result = self.scheduler.execute_operator(
            operation,
            inputs=dict(inputs or {}),
            batch=[dict(item) for item in (batch or [])],
            options=dict(options or {}),
            metadata=metadata,
        )
        return result.response

    def backbone(self, role: str = RUNTIME_ROLE_SCORER) -> SurfaceBackboneAdapter:
        key = str(role or RUNTIME_ROLE_SCORER)
        if key not in self._backbones:
            self._backbones[key] = SurfaceBackboneAdapter(self, role=key)
        return self._backbones[key]

    def embedding_client(self) -> SurfaceEmbeddingClient:
        if self._embedding_client is None:
            self._embedding_client = SurfaceEmbeddingClient(self)
        return self._embedding_client


__all__ = [
    "RuntimeInferenceContext",
    "SurfaceBackboneAdapter",
    "SurfaceCapabilities",
    "SurfaceEmbeddingClient",
    "normalize_oracle_config",
    "normalize_role_config",
    "normalize_runtime_inference_spec",
    "normalize_surface_config",
    "validate_public_runtime_config",
]
