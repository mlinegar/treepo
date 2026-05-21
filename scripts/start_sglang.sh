#!/bin/bash
# SGLang Server Launch Script for ThinkingTrees
# Reads model config from config/settings.yaml (reuses vllm.models for model paths)
#
# Usage:
#   ./scripts/start_sglang.sh                          # Uses default vllm model profile
#   ./scripts/start_sglang.sh nemotron-30b-nvfp4       # Uses specific model profile
#   ./scripts/start_sglang.sh qwen-80b --port 30001    # Override port
#
# SGLang serves the same OpenAI-compatible /v1/chat/completions endpoint as vLLM,
# so the same AsyncBatchLLMClient works with both backends.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

if [[ -z "${TT_START_ENGINE_DIRECT:-}" ]]; then
    exec python3 "$SCRIPT_DIR/start_engine.py" --engine sglang -- "$@"
fi

# Parse arguments
PROFILE=""
PORT_OVERRIDE=""
CUDA_DEVICES=""
TENSOR_PARALLEL_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
MEM_FRACTION_STATIC_OVERRIDE=""
SGLANG_VENV_PATH_OVERRIDE=""
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
SGLang Server Launcher

Usage:
  ./scripts/start_sglang.sh [PROFILE] [OPTIONS] [-- SGLANG_EXTRA_ARGS...]

Options:
  --port PORT                      Override server port
  --cuda-devices IDS               Set CUDA_VISIBLE_DEVICES (e.g. 0,1)
  --tensor-parallel N              Override tensor parallel size
  --max-model-len N                Override context length
  --context-length N               Alias for --max-model-len
  --mem-fraction-static RATIO      Override SGLang memory fraction
  --gpu-mem RATIO                  Alias for --mem-fraction-static
  --sglang-venv-path PATH          Override SGLang virtual environment
  -h, --help                       Show this help

Notes:
  - Unknown --flags are forwarded to SGLang as extra args.
  - PROFILE defaults to settings.yaml vllm.default.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --port)
            PORT_OVERRIDE="$2"
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
        --mem-fraction-static|--gpu-mem)
            MEM_FRACTION_STATIC_OVERRIDE="$2"
            shift 2
            ;;
        --sglang-venv-path)
            SGLANG_VENV_PATH_OVERRIDE="$2"
            shift 2
            ;;
        --*)
            # Collect unknown --flags as extra args to pass to SGLang
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

# Parse config with Python
read_config() {
    TT_CONFIG_FILE="$CONFIG_FILE" \
    TT_PROFILE="$PROFILE" \
    TT_SGLANG_VENV_PATH_OVERRIDE="$SGLANG_VENV_PATH_OVERRIDE" \
    python3 - <<'PY'
import os
import shlex
import sys

import yaml


def emit(key, value):
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = ""
    else:
        text = str(value)
    print(f"{key}={shlex.quote(text)}")


with open(os.environ["TT_CONFIG_FILE"]) as f:
    cfg = yaml.safe_load(f)

vllm = cfg.get("vllm", {})
models = vllm.get("models", {})
sglang_cfg = cfg.get("sglang", {})
inference_cfg = cfg.get("inference", {})
backend_cfg = inference_cfg.get("backend", {}) if isinstance(inference_cfg, dict) else {}
runtime = sglang_cfg.get("runtime", {}) if isinstance(sglang_cfg.get("runtime", {}), dict) else {}
runtime_overrides = runtime.get("profile_overrides", {}) if isinstance(runtime.get("profile_overrides", {}), dict) else {}

profile = os.environ.get("TT_PROFILE", "") or vllm.get("default", "small")

if profile not in models:
    print(
        f"ERROR: Profile '{profile}' not found. Available: {list(models.keys())}",
        file=sys.stderr,
    )
    sys.exit(1)

model_cfg = models[profile]
override = runtime_overrides.get(profile, {}) if isinstance(runtime_overrides.get(profile, {}), dict) else {}
effective_runtime = dict(runtime)
effective_runtime.update(override)
emit("PROFILE", profile)
emit("MODEL_PATH", model_cfg["path"])
emit("TENSOR_PARALLEL", model_cfg.get("tensor_parallel", 1))
emit("MAX_MODEL_LEN", model_cfg.get("max_model_len", 8192))

# SGLang-specific settings
emit("HOST", sglang_cfg.get("host", "0.0.0.0"))
emit("PORT", sglang_cfg.get("port", 30000))
emit("MEM_FRACTION_STATIC", sglang_cfg.get("mem_fraction_static", 0.88))

# Runtime tuning
emit("ENABLE_TORCH_COMPILE", bool(effective_runtime.get("enable_torch_compile", False)))
emit("CHUNKED_PREFILL_SIZE", effective_runtime.get("chunked_prefill_size", 0))
emit("DISABLE_RADIX_CACHE", bool(effective_runtime.get("disable_radix_cache", False)))
emit("ATTENTION_BACKEND", effective_runtime.get("attention_backend", ""))
emit("DISABLE_CUDA_GRAPH", bool(effective_runtime.get("disable_cuda_graph", False)))
emit("CUDA_GRAPH_MAX_BS", effective_runtime.get("cuda_graph_max_bs", 0))
emit(
    "SGLANG_VENV_PATH",
    os.environ.get("TT_SGLANG_VENV_PATH_OVERRIDE")
    or backend_cfg.get("sglang_venv_path", "${TREEPO_SGLANG_VENV:-$HOME/sglang-env}"),
)
emit("VLLM_VENV_PATH", backend_cfg.get("vllm_venv_path", "${TREEPO_VLLM_VENV:-$HOME/vllm-env}"))
PY
}

# Load config
CONFIG_VARS="$(read_config)" || exit 1
eval "$CONFIG_VARS"

# Activate the dedicated SGLang environment.
source "$SGLANG_VENV_PATH/bin/activate"

prepend_env_path() {
    local var_name="$1"
    local path_value="$2"
    local current="${!var_name:-}"

    if [[ -z "$path_value" ]]; then
        return
    fi
    if [[ ":$current:" == *":$path_value:"* ]]; then
        return
    fi
    if [[ -n "$current" ]]; then
        export "$var_name"="$path_value:$current"
    else
        export "$var_name"="$path_value"
    fi
}

nvcc_binary_works() {
    local nvcc_bin="$1"
    [[ -n "$nvcc_bin" && -x "$nvcc_bin" ]] || return 1
    "$nvcc_bin" --version >/dev/null 2>&1
}

configure_cuda_toolchain() {
    local toolkit_venv="$1"
    local cu13_root=""
    local cu13_nvcc=""
    local curand_include=""
    local current_nvcc=""
    local path_nvcc=""

    for candidate in "$toolkit_venv"/lib/python*/site-packages/nvidia/cu13; do
        if [[ -d "$candidate" ]]; then
            cu13_root="$candidate"
            break
        fi
    done
    for candidate in "$toolkit_venv"/lib/python*/site-packages/nvidia/curand/include; do
        if [[ -d "$candidate" ]]; then
            curand_include="$candidate"
            break
        fi
    done

    if [[ -n "${CUDA_HOME:-}" ]]; then
        current_nvcc="${CUDA_HOME}/bin/nvcc"
    fi
    if [[ -n "$cu13_root" ]]; then
        cu13_nvcc="${cu13_root}/bin/nvcc"
    fi

    # Prefer a toolkit root with a working nvcc --version invocation; some envs
    # expose a cuda_runtime path that looks executable but fails at runtime.
    if ! nvcc_binary_works "$current_nvcc"; then
        if nvcc_binary_works "$cu13_nvcc"; then
            export CUDA_HOME="$cu13_root"
        elif [[ -n "${CUDA_HOME:-}" ]]; then
            unset CUDA_HOME
        fi
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        prepend_env_path PATH "$CUDA_HOME/bin"
        prepend_env_path LD_LIBRARY_PATH "$CUDA_HOME/lib"
        prepend_env_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"
        export CUDA_PATH="$CUDA_HOME"
    else
        unset CUDA_PATH
    fi
    if [[ -n "${CUDA_HOME:-}" ]] && nvcc_binary_works "${CUDA_HOME}/bin/nvcc"; then
        export FLASHINFER_NVCC="${CUDA_HOME}/bin/nvcc"
        export CUDACXX="${CUDA_HOME}/bin/nvcc"
    else
        path_nvcc="$(command -v nvcc 2>/dev/null || true)"
        if nvcc_binary_works "$path_nvcc"; then
            export FLASHINFER_NVCC="$path_nvcc"
            export CUDACXX="$path_nvcc"
        else
            unset FLASHINFER_NVCC
            unset CUDACXX
        fi
    fi
    prepend_env_path LD_LIBRARY_PATH "/lib/x86_64-linux-gnu"
    if [[ -n "$curand_include" ]]; then
        prepend_env_path CPATH "$curand_include"
    fi
}

configure_flashinfer_workspace() {
    if [[ -n "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
        return
    fi

    local profile_name=""
    local profile_slug=""
    local key=""
    local digest=""

    profile_name="$(basename "${MODEL_PATH:-}")"
    if [[ -z "$profile_name" || "$profile_name" == "/" || "$profile_name" == "." ]]; then
        profile_name="$PROFILE"
    fi
    profile_slug="$(printf '%s' "$profile_name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
    profile_slug="${profile_slug:0:48}"
    if [[ -z "$profile_slug" ]]; then
        profile_slug="model"
    fi

    key="${VLLM_VENV_PATH}|${CUDA_HOME:-}|${FLASHINFER_NVCC:-}|${MODEL_PATH}|${PROFILE}"
    if command -v sha1sum >/dev/null 2>&1; then
        digest="$(printf '%s' "$key" | sha1sum | awk '{print substr($1,1,12)}')"
    else
        digest="$(printf '%s' "$key" | cksum | awk '{print $1}')"
    fi
    if [[ -z "$digest" ]]; then
        digest="default"
    fi

    export FLASHINFER_WORKSPACE_BASE="/tmp/thinkingtrees/flashinfer/${profile_slug}-${digest}"
    mkdir -p "$FLASHINFER_WORKSPACE_BASE" >/dev/null 2>&1 || true
}

configure_cuda_toolchain "$VLLM_VENV_PATH"

if [[ "$PROFILE" == *"nvfp4"* || "$MODEL_PATH" == *"NVFP4"* || "$MODEL_PATH" == *"nvfp4"* ]]; then
    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
    export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"
    configure_flashinfer_workspace
fi

# Apply command-line overrides
if [[ -n "$PORT_OVERRIDE" ]]; then
    PORT="$PORT_OVERRIDE"
fi
if [[ -n "$TENSOR_PARALLEL_OVERRIDE" ]]; then
    TENSOR_PARALLEL="$TENSOR_PARALLEL_OVERRIDE"
fi
if [[ -n "$MAX_MODEL_LEN_OVERRIDE" ]]; then
    MAX_MODEL_LEN="$MAX_MODEL_LEN_OVERRIDE"
fi
if [[ -n "$MEM_FRACTION_STATIC_OVERRIDE" ]]; then
    MEM_FRACTION_STATIC="$MEM_FRACTION_STATIC_OVERRIDE"
fi

echo "=========================================="
echo "Starting SGLang Server"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Mem Fraction Static: $MEM_FRACTION_STATIC"
if [[ "$ENABLE_TORCH_COMPILE" == "true" ]]; then
    echo "Torch Compile: ENABLED"
fi
if [[ "$DISABLE_RADIX_CACHE" == "true" ]]; then
    echo "Radix Cache: DISABLED"
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
    echo "Attention Backend: $ATTENTION_BACKEND"
fi
if [[ "$DISABLE_CUDA_GRAPH" == "true" ]]; then
    echo "CUDA Graph: DISABLED"
fi
if [[ "$PROFILE" == *"nvfp4"* || "$MODEL_PATH" == *"NVFP4"* || "$MODEL_PATH" == *"nvfp4"* ]]; then
    echo "CUDA_HOME: ${CUDA_HOME:-unset}"
    echo "FLASHINFER_NVCC: ${FLASHINFER_NVCC:-unset}"
    echo "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE:-unset}"
    echo "Quantization Override: modelopt_fp4"
fi
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "CUDA Devices: $CUDA_DEVICES"
fi
echo "=========================================="

# Set CUDA device isolation if specified
if [[ -n "$CUDA_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# Build SGLang command
SGLANG_CMD=(
    python -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --tp "$TENSOR_PARALLEL"
    --context-length "$MAX_MODEL_LEN"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --trust-remote-code
)

if [[ "$PROFILE" == *"nvfp4"* || "$MODEL_PATH" == *"NVFP4"* || "$MODEL_PATH" == *"nvfp4"* ]]; then
    SGLANG_CMD+=(--quantization modelopt_fp4)
fi

# Optional runtime flags
if [[ "$ENABLE_TORCH_COMPILE" == "true" ]]; then
    SGLANG_CMD+=(--enable-torch-compile)
fi

if [[ "$CHUNKED_PREFILL_SIZE" != "0" && -n "$CHUNKED_PREFILL_SIZE" ]]; then
    SGLANG_CMD+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
fi

if [[ "$DISABLE_RADIX_CACHE" == "true" ]]; then
    SGLANG_CMD+=(--disable-radix-cache)
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
    SGLANG_CMD+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ "$DISABLE_CUDA_GRAPH" == "true" ]]; then
    SGLANG_CMD+=(--disable-cuda-graph)
fi
if [[ "$CUDA_GRAPH_MAX_BS" != "0" && -n "$CUDA_GRAPH_MAX_BS" ]]; then
    SGLANG_CMD+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi

# Launch SGLang server
"${SGLANG_CMD[@]}" "${EXTRA_ARGS[@]}"
