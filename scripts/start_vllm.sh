#!/bin/bash
# vLLM server launch helper for treepo-compatible OpenAI endpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${TREEPO_SETTINGS:-$PROJECT_ROOT/config/settings.yaml}"

PROFILE=""
MODEL_OVERRIDE=""
SERVED_MODEL_NAME_OVERRIDE=""
HOST_OVERRIDE=""
PORT_OVERRIDE=""
CUDA_DEVICES=""
TENSOR_PARALLEL_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
GPU_MEM_OVERRIDE=""
API_KEY_OVERRIDE=""
CONFIG_OVERRIDE=""
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
vLLM Server Launcher

Usage:
  ./scripts/start_vllm.sh --model PATH_OR_HF_ID --served-model-name MODEL_ID [OPTIONS]
  ./scripts/start_vllm.sh [PROFILE] [OPTIONS]

Options:
  --config PATH                 Settings YAML. Defaults to config/settings.yaml.
  --model PATH_OR_HF_ID         Launch this model directly instead of reading a profile.
  --served-model-name NAME      Model id exposed by /v1/models.
  --host HOST                   Server host, default 0.0.0.0.
  --port PORT                   Server port, default 8000.
  --api-key KEY                 API key required by the OpenAI-compatible server.
  --cuda-devices IDS            Set CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1.
  --tensor-parallel N           Override tensor parallel size.
  --max-model-len N             Override max model length.
  --context-length N            Alias for --max-model-len.
  --gpu-mem RATIO               Override --gpu-memory-utilization.
  -h, --help                    Show this help.

Unknown --flags after the profile/options are forwarded to vLLM. Direct model
launches do not require any config file.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --config)
            CONFIG_OVERRIDE="$2"
            shift 2
            ;;
        --model)
            MODEL_OVERRIDE="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME_OVERRIDE="$2"
            shift 2
            ;;
        --host)
            HOST_OVERRIDE="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --api-key)
            API_KEY_OVERRIDE="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --tensor-parallel)
            TENSOR_PARALLEL_OVERRIDE="$2"
            shift 2
            ;;
        --max-model-len|--context-length)
            MAX_MODEL_LEN_OVERRIDE="$2"
            shift 2
            ;;
        --gpu-mem)
            GPU_MEM_OVERRIDE="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        --*)
            EXTRA_ARGS+=("$1")
            if [[ $# -gt 1 && ! "$2" =~ ^-- ]]; then
                EXTRA_ARGS+=("$2")
                shift
            fi
            shift
            ;;
        *)
            PROFILE="$1"
            shift
            ;;
    esac
done

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    CONFIG_FILE="$CONFIG_OVERRIDE"
fi

VLLM_VENV="${TREEPO_VLLM_VENV:-$HOME/vllm-env}"
if [[ -d "$VLLM_VENV" ]]; then
    # shellcheck source=/dev/null
    source "$VLLM_VENV/bin/activate"
fi

prepend_env_path() {
    local var_name="$1"
    local path_value="$2"
    local current="${!var_name:-}"

    [[ -n "$path_value" && -d "$path_value" ]] || return
    if [[ ":$current:" == *":$path_value:"* ]]; then
        return
    fi
    if [[ -n "$current" ]]; then
        export "$var_name"="$path_value:$current"
    else
        export "$var_name"="$path_value"
    fi
}

ensure_soname_symlink() {
    local base_dir="$1"
    local link_name="$2"
    local target_name="$3"

    if [[ -d "$base_dir" && -e "$base_dir/$target_name" && ! -e "$base_dir/$link_name" ]]; then
        ln -s "$target_name" "$base_dir/$link_name" >/dev/null 2>&1 || true
    fi
}

find_python_package_dir() {
    local suffix="$1"
    python3 - "$suffix" <<'PY'
import os
import site
import sys

suffix = sys.argv[1]
roots = []
for fn in (site.getsitepackages,):
    try:
        roots.extend(fn())
    except Exception:
        pass
try:
    roots.append(site.getusersitepackages())
except Exception:
    pass
for root in roots:
    candidate = os.path.join(root, suffix)
    if os.path.isdir(candidate):
        print(candidate)
        break
PY
}

configure_python_cuda_runtime() {
    local cu_root=""
    local curand_include=""

    if [[ -z "${CUDA_HOME:-}" ]]; then
        cu_root="$(find_python_package_dir "nvidia/cu13" || true)"
        if [[ -n "$cu_root" ]]; then
            export CUDA_HOME="$cu_root"
        fi
    fi
    curand_include="$(find_python_package_dir "nvidia/curand/include" || true)"

    if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
        mkdir -p "$CUDA_HOME/lib64" "$CUDA_HOME/lib64/stubs" >/dev/null 2>&1 || true
        ensure_soname_symlink "$CUDA_HOME/lib" "libcudart.so" "libcudart.so.13"
        ensure_soname_symlink "$CUDA_HOME/lib" "libcublas.so" "libcublas.so.13"
        ensure_soname_symlink "$CUDA_HOME/lib" "libcublasLt.so" "libcublasLt.so.13"
        ensure_soname_symlink "$CUDA_HOME/lib" "libnvrtc.so" "libnvrtc.so.13"
        if [[ -d "$CUDA_HOME/lib" ]]; then
            [[ -e "$CUDA_HOME/lib64/libcudart.so" ]] || ln -s ../lib/libcudart.so "$CUDA_HOME/lib64/libcudart.so" >/dev/null 2>&1 || true
            [[ -e "$CUDA_HOME/lib64/libcudart.so.13" ]] || ln -s ../lib/libcudart.so.13 "$CUDA_HOME/lib64/libcudart.so.13" >/dev/null 2>&1 || true
            [[ -e "$CUDA_HOME/lib64/libcublas.so" ]] || ln -s ../lib/libcublas.so "$CUDA_HOME/lib64/libcublas.so" >/dev/null 2>&1 || true
            [[ -e "$CUDA_HOME/lib64/libcublasLt.so" ]] || ln -s ../lib/libcublasLt.so "$CUDA_HOME/lib64/libcublasLt.so" >/dev/null 2>&1 || true
            [[ -e "$CUDA_HOME/lib64/libnvrtc.so" ]] || ln -s ../lib/libnvrtc.so "$CUDA_HOME/lib64/libnvrtc.so" >/dev/null 2>&1 || true
        fi
        if [[ -e /lib/x86_64-linux-gnu/libcuda.so && ! -e "$CUDA_HOME/lib64/stubs/libcuda.so" ]]; then
            ln -s /lib/x86_64-linux-gnu/libcuda.so "$CUDA_HOME/lib64/stubs/libcuda.so" >/dev/null 2>&1 || true
        fi
        prepend_env_path PATH "$CUDA_HOME/bin"
        prepend_env_path LD_LIBRARY_PATH "$CUDA_HOME/lib"
        prepend_env_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"
        prepend_env_path LD_LIBRARY_PATH /lib/x86_64-linux-gnu
    fi
    if [[ -n "$curand_include" ]]; then
        prepend_env_path CPATH "$curand_include"
    fi
}

read_profile_config() {
    TT_CONFIG_FILE="$CONFIG_FILE" TT_PROFILE="$PROFILE" python3 - <<'PY'
import json
import os
import shlex
import sys

try:
    import yaml
except ImportError as exc:
    raise SystemExit("PyYAML is required to read config/settings.yaml.") from exc


def emit(key, value):
    print(f"{key}={shlex.quote(str(value))}")


path = os.environ["TT_CONFIG_FILE"]
profile = os.environ.get("TT_PROFILE") or ""
with open(path, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
vllm = cfg.get("vllm") or {}
models = vllm.get("models") or {}
profile = profile or vllm.get("default") or ""
if profile not in models:
    raise SystemExit(f"Profile {profile!r} not found in {path}. Available: {sorted(models)}")

model_cfg = models[profile] or {}
runtime = vllm.get("runtime") if isinstance(vllm.get("runtime"), dict) else {}
profile_overrides = runtime.get("profile_overrides") if isinstance(runtime.get("profile_overrides"), dict) else {}
profile_runtime = profile_overrides.get(profile) if isinstance(profile_overrides.get(profile), dict) else {}
flags = []
for source in (runtime, profile_runtime, model_cfg):
    extra = source.get("extra_flags") if isinstance(source, dict) else None
    if isinstance(extra, list):
        flags.extend(str(x) for x in extra)

emit("PROFILE", profile)
emit("MODEL_PATH", model_cfg.get("path") or "")
emit("SERVED_MODEL_NAME", model_cfg.get("served_model_name") or "")
emit("HOST", vllm.get("host", "0.0.0.0"))
emit("PORT", vllm.get("port", 8000))
emit("TENSOR_PARALLEL", model_cfg.get("tensor_parallel", 1))
emit("MAX_MODEL_LEN", model_cfg.get("max_model_len", 8192))
emit("GPU_MEM", model_cfg.get("gpu_memory_utilization", vllm.get("gpu_memory_utilization", 0.90)))
emit("API_KEY", vllm.get("api_key", ""))
emit("PREFIX_CACHE", bool(vllm.get("enable_prefix_caching", False)).__str__().lower())
emit("RUNTIME_ARGS_JSON", json.dumps(flags))
PY
}

configure_python_cuda_runtime

if [[ -n "$MODEL_OVERRIDE" ]]; then
    PROFILE="${PROFILE:-direct}"
    MODEL_PATH="$MODEL_OVERRIDE"
    SERVED_MODEL_NAME="$SERVED_MODEL_NAME_OVERRIDE"
    HOST="${HOST_OVERRIDE:-0.0.0.0}"
    PORT="${PORT_OVERRIDE:-8000}"
    TENSOR_PARALLEL="${TENSOR_PARALLEL_OVERRIDE:-1}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-8192}"
    GPU_MEM="${GPU_MEM_OVERRIDE:-0.90}"
    API_KEY="$API_KEY_OVERRIDE"
    PREFIX_CACHE="false"
    RUNTIME_ARGS=()
else
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: no --model supplied and settings file not found: $CONFIG_FILE" >&2
        echo "Use --model PATH_OR_HF_ID or copy config/settings.example.yaml to config/settings.yaml." >&2
        exit 1
    fi
    CONFIG_VARS="$(read_profile_config)" || exit 1
    eval "$CONFIG_VARS"
    if [[ -n "$HOST_OVERRIDE" ]]; then HOST="$HOST_OVERRIDE"; fi
    if [[ -n "$PORT_OVERRIDE" ]]; then PORT="$PORT_OVERRIDE"; fi
    if [[ -n "$SERVED_MODEL_NAME_OVERRIDE" ]]; then SERVED_MODEL_NAME="$SERVED_MODEL_NAME_OVERRIDE"; fi
    if [[ -n "$TENSOR_PARALLEL_OVERRIDE" ]]; then TENSOR_PARALLEL="$TENSOR_PARALLEL_OVERRIDE"; fi
    if [[ -n "$MAX_MODEL_LEN_OVERRIDE" ]]; then MAX_MODEL_LEN="$MAX_MODEL_LEN_OVERRIDE"; fi
    if [[ -n "$GPU_MEM_OVERRIDE" ]]; then GPU_MEM="$GPU_MEM_OVERRIDE"; fi
    if [[ -n "$API_KEY_OVERRIDE" ]]; then API_KEY="$API_KEY_OVERRIDE"; fi
    RUNTIME_ARGS=()
    if [[ -n "${RUNTIME_ARGS_JSON:-}" ]]; then
        mapfile -t RUNTIME_ARGS < <(RUNTIME_ARGS_JSON="$RUNTIME_ARGS_JSON" python3 - <<'PY'
import json
import os
for item in json.loads(os.environ.get("RUNTIME_ARGS_JSON") or "[]"):
    print(item)
PY
)
    fi
fi

model_lc="$(printf '%s %s' "${PROFILE:-}" "${MODEL_PATH:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "$model_lc" == *"nvfp4"* ]]; then
    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
    export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"
    if [[ -z "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
        slug="$(basename "${MODEL_PATH:-$PROFILE}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
        export FLASHINFER_WORKSPACE_BASE="/tmp/treepo/flashinfer/${slug:-model}"
        mkdir -p "$FLASHINFER_WORKSPACE_BASE" >/dev/null 2>&1 || true
    fi
fi

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: resolved model path is empty." >&2
    exit 1
fi
if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* ]]; then
    if [[ ! -e "$MODEL_PATH" ]]; then
        echo "ERROR: model path not found: $MODEL_PATH" >&2
        exit 1
    fi
fi
if [[ -n "$CUDA_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Profile: ${PROFILE:-direct}"
echo "Model: $MODEL_PATH"
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then echo "Served Model Name: $SERVED_MODEL_NAME"; fi
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEM"
if [[ -n "${CUDA_HOME:-}" ]]; then echo "CUDA_HOME: $CUDA_HOME"; fi
if [[ -n "$CUDA_DEVICES" ]]; then echo "CUDA Devices: $CUDA_DEVICES"; fi
echo "=========================================="

VLLM_CMD=(
    python -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEM"
    --trust-remote-code
)
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
    VLLM_CMD+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ -n "${API_KEY:-}" ]]; then
    VLLM_CMD+=(--api-key "$API_KEY")
fi
if [[ "$PREFIX_CACHE" == "true" ]]; then
    VLLM_CMD+=(--enable-prefix-caching)
fi
if [[ ${#RUNTIME_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${RUNTIME_ARGS[@]}")
fi

"${VLLM_CMD[@]}" "${EXTRA_ARGS[@]}"
