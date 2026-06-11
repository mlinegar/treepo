#!/bin/bash
# vLLM Server Launch Script for ThinkingTrees
# Reads model config from config/settings.yaml
#
# Usage:
#   ./scripts/start_vllm.sh                    # Uses default model from config
#   ./scripts/start_vllm.sh qwen-80b           # Uses "qwen-80b" model profile
#   ./scripts/start_vllm.sh --preset training  # Uses speculative decoding preset
#   ./scripts/start_vllm.sh --preset inference # Uses inference preset (small model only)
#
# Speculative Decoding:
#   When a preset with speculative decoding is used, the server runs with:
#   - Target model: the main model for verification
#   - Draft model: smaller model for fast token proposal
#   - Result: 1.5-3x faster generation with identical outputs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/settings.yaml"

if [[ -z "${TT_START_ENGINE_DIRECT:-}" ]]; then
    exec python3 "$SCRIPT_DIR/start_engine.py" --engine vllm -- "$@"
fi

# Parse arguments
PROFILE=""
PRESET=""
PORT_OVERRIDE=""
KV_CACHE_DTYPE=""
CUDA_DEVICES=""
TENSOR_PARALLEL_OVERRIDE=""
MAX_MODEL_LEN_OVERRIDE=""
GPU_MEM_OVERRIDE=""
EXTRA_ARGS=()

show_help() {
    cat <<'EOF'
vLLM Server Launcher

Usage:
  ./scripts/start_vllm.sh [PROFILE] [OPTIONS] [-- VLLM_EXTRA_ARGS...]

Options:
  --preset NAME                Use speculative preset from config
  --port PORT                  Override server port
  --kv-cache-dtype DTYPE       Set --kv-cache-dtype for vLLM
  --cuda-devices IDS           Set CUDA_VISIBLE_DEVICES (e.g. 0,1)
  --tensor-parallel N          Override tensor parallel size
  --max-model-len N            Override max model length
  --context-length N           Alias for --max-model-len
  --gpu-mem RATIO              Override --gpu-memory-utilization
  -h, --help                   Show this help

Notes:
  - Unknown --flags are forwarded to vLLM as extra args.
  - PROFILE defaults to settings.yaml vllm.default.
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --port)
            PORT_OVERRIDE="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
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
        --*)
            # Collect unknown --flags as extra args to pass to vLLM
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

# Activate the vllm environment
source ${TREEPO_VLLM_VENV:-$HOME/vllm-env}/bin/activate

# Parse config with Python (more reliable than shell YAML parsing)
read_config() {
    TT_CONFIG_FILE="$CONFIG_FILE" \
    TT_PRESET="$PRESET" \
    TT_PROFILE="$PROFILE" \
    TT_PROJECT_ROOT="$PROJECT_ROOT" \
    python3 - <<'PY'
import os
import shlex
import sys
import importlib.util

import yaml

runtime_module_path = os.path.join(
    os.environ["TT_PROJECT_ROOT"],
    "src",
    "core",
    "vllm_runtime.py",
)
runtime_spec = importlib.util.spec_from_file_location("tt_vllm_runtime", runtime_module_path)
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError(f"Could not load runtime helper module: {runtime_module_path}")
runtime_module = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime_module
runtime_spec.loader.exec_module(runtime_module)
resolve_vllm_runtime_flags = runtime_module.resolve_vllm_runtime_flags


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
speculative = cfg.get("speculative", {})
presets = speculative.get("presets", {})

preset_name = os.environ.get("TT_PRESET", "")
profile = os.environ.get("TT_PROFILE", "")
resolved_profile = ""

# If preset specified, use it to determine target and draft models
if preset_name:
    if preset_name not in presets:
        print(
            f"ERROR: Preset '{preset_name}' not found. Available: {list(presets.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    preset = presets[preset_name]
    target_profile = preset["target"]
    draft_profile = preset.get("draft")
    spec_enabled = preset.get("enabled", False)
    num_spec_tokens = preset.get(
        "num_speculative_tokens",
        speculative.get("num_speculative_tokens", 5),
    )

    if target_profile not in models:
        print(
            f"ERROR: Target model '{target_profile}' not found. Available: {list(models.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    target = models[target_profile]
    resolved_profile = str(target_profile)
    emit("PROFILE", target_profile)
    emit("MODEL_PATH", target["path"])
    emit("SERVED_MODEL_NAME", target.get("served_model_name", ""))
    emit("TENSOR_PARALLEL", target.get("tensor_parallel", 1))
    emit("MAX_MODEL_LEN", target.get("max_model_len", 8192))

    # Speculative decoding settings
    if spec_enabled and draft_profile:
        if draft_profile not in models:
            print(
                f"ERROR: Draft model '{draft_profile}' not found. Available: {list(models.keys())}",
                file=sys.stderr,
            )
            sys.exit(1)
        draft = models[draft_profile]
        emit("SPEC_ENABLED", True)
        emit("DRAFT_MODEL_PATH", draft["path"])
        emit("DRAFT_TENSOR_PARALLEL", draft.get("tensor_parallel", 1))
        emit("NUM_SPEC_TOKENS", num_spec_tokens)
    else:
        emit("SPEC_ENABLED", False)
else:
    # No preset - use profile directly (backwards compatible)
    profile = profile or vllm.get("default", "small")

    if profile not in models:
        print(
            f"ERROR: Profile '{profile}' not found. Available: {list(models.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    model_cfg = models[profile]
    resolved_profile = str(profile)
    emit("PROFILE", profile)
    emit("MODEL_PATH", model_cfg["path"])
    emit("SERVED_MODEL_NAME", model_cfg.get("served_model_name", ""))
    emit("TENSOR_PARALLEL", model_cfg.get("tensor_parallel", 1))
    emit("MAX_MODEL_LEN", model_cfg.get("max_model_len", 8192))
    emit("SPEC_ENABLED", False)

runtime = resolve_vllm_runtime_flags(vllm_cfg=vllm, profile=resolved_profile)

emit("HOST", vllm.get("host", "0.0.0.0"))
emit("PORT", vllm.get("port", 8000))
emit("GPU_MEM", vllm.get("gpu_memory_utilization", 0.90))
emit("PREFIX_CACHE", bool(vllm.get("enable_prefix_caching", False)))
emit("RUNTIME_ARGS_JOINED", "\x1f".join(runtime.to_cli_args()))
PY
}

# Load config
CONFIG_VARS="$(read_config)" || exit 1
eval "$CONFIG_VARS"

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
if [[ -n "$GPU_MEM_OVERRIDE" ]]; then
    GPU_MEM="$GPU_MEM_OVERRIDE"
fi
RUNTIME_ARGS=()
if [[ -n "$RUNTIME_ARGS_JOINED" ]]; then
    IFS=$'\x1f' read -r -a RUNTIME_ARGS <<< "$RUNTIME_ARGS_JOINED"
fi

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

ensure_soname_symlink() {
    local base_dir="$1"
    local link_name="$2"
    local target_name="$3"

    if [[ -z "$base_dir" || -z "$link_name" || -z "$target_name" ]]; then
        return
    fi
    if [[ ! -d "$base_dir" ]]; then
        return
    fi
    if [[ -e "$base_dir/$target_name" && ! -e "$base_dir/$link_name" ]]; then
        ln -s "$target_name" "$base_dir/$link_name" >/dev/null 2>&1 || true
    fi
}

configure_nvfp4_runtime() {
    local profile_lc=""
    local model_path_lc=""
    profile_lc="$(printf '%s' "${PROFILE:-}" | tr '[:upper:]' '[:lower:]')"
    model_path_lc="$(printf '%s' "${MODEL_PATH:-}" | tr '[:upper:]' '[:lower:]')"
    if [[ "$profile_lc" != *"nvfp4"* && "$model_path_lc" != *"nvfp4"* ]]; then
        return
    fi

    export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-1}"
    export VLLM_FLASHINFER_MOE_BACKEND="${VLLM_FLASHINFER_MOE_BACKEND:-throughput}"

    local site_packages
    site_packages="$(python3 - <<'PY'
import site
for p in site.getsitepackages():
    if "site-packages" in p:
        print(p)
        break
PY
)"
    local cu13_root=""
    local curand_include=""
    if [[ -n "$site_packages" ]]; then
        cu13_root="${site_packages}/nvidia/cu13"
        curand_include="${site_packages}/nvidia/curand/include"
    fi

    # Prefer the uv-managed CUDA toolchain if system nvcc is unavailable.
    if ! command -v nvcc >/dev/null 2>&1; then
        if [[ -z "${CUDA_HOME:-}" && -x "${cu13_root}/bin/nvcc" ]]; then
            export CUDA_HOME="${cu13_root}"
        fi
        if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
            prepend_env_path PATH "${CUDA_HOME}/bin"
        fi
    fi

    if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}" ]]; then
        mkdir -p "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib64/stubs"

        ensure_soname_symlink "${CUDA_HOME}/lib" "libcudart.so" "libcudart.so.13"
        ensure_soname_symlink "${CUDA_HOME}/lib" "libcublas.so" "libcublas.so.13"
        ensure_soname_symlink "${CUDA_HOME}/lib" "libcublasLt.so" "libcublasLt.so.13"
        ensure_soname_symlink "${CUDA_HOME}/lib" "libnvrtc.so" "libnvrtc.so.13"

        if [[ -d "${CUDA_HOME}/lib" ]]; then
            [[ -e "${CUDA_HOME}/lib64/libcudart.so" ]] || ln -s ../lib/libcudart.so "${CUDA_HOME}/lib64/libcudart.so" >/dev/null 2>&1 || true
            [[ -e "${CUDA_HOME}/lib64/libcudart.so.13" ]] || ln -s ../lib/libcudart.so.13 "${CUDA_HOME}/lib64/libcudart.so.13" >/dev/null 2>&1 || true
            [[ -e "${CUDA_HOME}/lib64/libcublas.so" ]] || ln -s ../lib/libcublas.so "${CUDA_HOME}/lib64/libcublas.so" >/dev/null 2>&1 || true
            [[ -e "${CUDA_HOME}/lib64/libcublasLt.so" ]] || ln -s ../lib/libcublasLt.so "${CUDA_HOME}/lib64/libcublasLt.so" >/dev/null 2>&1 || true
            [[ -e "${CUDA_HOME}/lib64/libnvrtc.so" ]] || ln -s ../lib/libnvrtc.so "${CUDA_HOME}/lib64/libnvrtc.so" >/dev/null 2>&1 || true
        fi
        if [[ -e /lib/x86_64-linux-gnu/libcuda.so && ! -e "${CUDA_HOME}/lib64/stubs/libcuda.so" ]]; then
            ln -s /lib/x86_64-linux-gnu/libcuda.so "${CUDA_HOME}/lib64/stubs/libcuda.so" >/dev/null 2>&1 || true
        fi

        prepend_env_path LD_LIBRARY_PATH "${CUDA_HOME}/lib"
        prepend_env_path LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
        prepend_env_path LD_LIBRARY_PATH /lib/x86_64-linux-gnu
    fi

    if [[ -d "${curand_include}" ]]; then
        prepend_env_path CPATH "${curand_include}"
    fi

    local missing=()
    if ! command -v nvcc >/dev/null 2>&1; then
        missing+=("nvcc")
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        [[ -f "${CUDA_HOME}/include/cublasLt.h" ]] || missing+=("cublasLt.h")
        [[ -f "${CUDA_HOME}/lib/libnvrtc.so.13" || -f "${CUDA_HOME}/lib/libnvrtc.so" ]] || missing+=("libnvrtc")
    fi
    if [[ -n "${curand_include}" ]]; then
        [[ -f "${curand_include}/curand_kernel.h" ]] || missing+=("curand_kernel.h")
    fi
    if (( ${#missing[@]} > 0 )); then
        echo "WARNING: NVFP4 prerequisites missing: ${missing[*]}"
        echo "Install in ${TREEPO_VLLM_VENV:-$HOME/vllm-env}:"
        echo "  uv pip install --python ${TREEPO_VLLM_VENV:-$HOME/vllm-env}/bin/python flashinfer-cubin==0.5.3 nvidia-cuda-nvcc==13.1.115 nvidia-cuda-cccl==13.1.115 nvidia-cublas==13.1.0.3 nvidia-cuda-nvrtc==13.1.115"
    fi

    if [[ -z "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
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

        key="${VIRTUAL_ENV:-}|${CUDA_HOME:-}|${FLASHINFER_NVCC:-}|${MODEL_PATH}|${PROFILE}"
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
    fi
}

configure_nvfp4_runtime

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Profile: $PROFILE"
echo "Model: $MODEL_PATH"
if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
    echo "Served Model Name: ${SERVED_MODEL_NAME}"
fi
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Prefix Cache: $PREFIX_CACHE"
if [[ ${#RUNTIME_ARGS[@]} -gt 0 ]]; then
    echo "Runtime Flags: ${RUNTIME_ARGS[*]}"
fi
if [[ "$(printf '%s' "${PROFILE:-}" | tr '[:upper:]' '[:lower:]')" == *"nvfp4"* || "$(printf '%s' "${MODEL_PATH:-}" | tr '[:upper:]' '[:lower:]')" == *"nvfp4"* ]]; then
    echo "NVFP4 FlashInfer FP4: VLLM_USE_FLASHINFER_MOE_FP4=${VLLM_USE_FLASHINFER_MOE_FP4:-unset} VLLM_FLASHINFER_MOE_BACKEND=${VLLM_FLASHINFER_MOE_BACKEND:-unset}"
    if [[ -n "${CUDA_HOME:-}" ]]; then
        echo "CUDA_HOME: ${CUDA_HOME}"
    fi
    echo "FLASHINFER_WORKSPACE_BASE: ${FLASHINFER_WORKSPACE_BASE:-unset}"
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    echo "KV Cache DType: $KV_CACHE_DTYPE"
fi
if [[ "$SPEC_ENABLED" == "true" ]]; then
    echo "------------------------------------------"
    echo "Speculative Decoding: ENABLED"
    echo "Draft Model: $DRAFT_MODEL_PATH"
    echo "Draft Tensor Parallel: $DRAFT_TENSOR_PARALLEL"
    echo "Speculative Tokens: $NUM_SPEC_TOKENS"
fi
if [[ -n "$CUDA_DEVICES" ]]; then
    echo "CUDA Devices: $CUDA_DEVICES"
fi
echo "=========================================="

# Helpful preflight: if MODEL_PATH is a local path, verify it exists.
if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* ]]; then
    if [[ ! -e "$MODEL_PATH" ]]; then
        echo "ERROR: Model path not found: $MODEL_PATH" 1>&2
        if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
            echo "Download with:" 1>&2
            echo "  ./scripts/download_hf_model.sh \"${SERVED_MODEL_NAME}\" \"${MODEL_PATH}\"" 1>&2
        else
            echo "Download with:" 1>&2
            echo "  ./scripts/download_hf_model.sh <hf_model_id> \"${MODEL_PATH}\"" 1>&2
        fi
        exit 1
    fi
fi

# Set CUDA device isolation if specified
if [[ -n "$CUDA_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

# Build vLLM command
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
    VLLM_CMD+=(--served-model-name "${SERVED_MODEL_NAME}")
fi

# Add prefix caching unless explicitly overridden by runtime flags or CLI args.
PREFIX_FLAG_PRESENT=false
for arg in "${RUNTIME_ARGS[@]}" "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--enable-prefix-caching" || "$arg" == "--no-enable-prefix-caching" || "$arg" == "--disable-prefix-caching" ]]; then
        PREFIX_FLAG_PRESENT=true
        break
    fi
done
if [[ "$PREFIX_CACHE" == "true" && "$PREFIX_FLAG_PRESENT" == "false" ]]; then
    VLLM_CMD+=(--enable-prefix-caching)
fi

# Add runtime tuning flags from settings (profile-aware).
if [[ ${#RUNTIME_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${RUNTIME_ARGS[@]}")
fi

# Add --kv-cache-dtype if specified
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    VLLM_CMD+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

# Add speculative decoding flags if enabled
if [[ "$SPEC_ENABLED" == "true" ]]; then
    VLLM_CMD+=(
        --speculative-model "$DRAFT_MODEL_PATH"
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
        --speculative-draft-tensor-parallel-size "$DRAFT_TENSOR_PARALLEL"
    )
fi

# Launch vLLM server
"${VLLM_CMD[@]}" "${EXTRA_ARGS[@]}"
