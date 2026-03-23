# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorRT-LLM is NVIDIA's library for optimized LLM inference on NVIDIA GPUs. It provides a Python API to define LLMs and a high-performance C++ runtime for inference execution. The project is a hybrid Python/C++ codebase with CUDA kernels.

## Build Commands

### Development Container (recommended)

```bash
# Start the dev container (mount local source at /code/tensorrt_llm)
make -C docker ngc-devel_run LOCAL_USER=1 DOCKER_PULL=1 IMAGE_TAG=x.y.z
```

### Building the Wheel

```bash
# Full build (inside dev container or with proper deps)
./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=native

# Install the built wheel
pip install ./build/tensorrt_llm*.whl

# For editable install (must build first)
./scripts/build_wheel.py --clean --use_ccache --cuda_architectures=native
pip install -e .
```

The C++ build system uses CMake (via `cpp/CMakeLists.txt`). The `build_wheel.py` script handles both C++ compilation and Python wheel packaging. Key build flags: `--clean`, `--use_ccache`, `--cuda_architectures=native`. Use `--help` for all options.

Build output goes to `cpp/build/` (Release) or `cpp/build_<type>/` for other build types.

### Pre-commit / Linting

```bash
pip install pre-commit
pre-commit install
# Runs automatically on commit; includes: isort, yapf, autoflake, clang-format, cmake-format, codespell, ruff, ruff-format, mdformat
```

Legacy files use isort/yapf/autoflake; newer code (particularly `tensorrt_llm/_torch/auto_deploy/`) uses ruff. The project is migrating toward ruff.

## Testing

Tests are in `tests/` with three main categories:
- `tests/unittest/` — Unit tests (pytest-based)
- `tests/integration/` — Integration tests
- `tests/microbenchmarks/` — Performance benchmarks

```bash
# Run a single unit test file
pytest tests/unittest/<path_to_test>.py

# Run a specific test
pytest tests/unittest/<path_to_test>.py::TestClass::test_name

# Run with verbose output
pytest tests/unittest/<path_to_test>.py -v
```

Test markers (defined in `tests/unittest/pytest.ini`): `gpu2`, `gpu4`, `post_merge`, `high_cuda_memory`, `no_xdist`, `ray`.

C++ tests are built via CMake (`BUILD_TESTS=ON`) and located in `cpp/tests/`.

## Architecture

### Python Package (`tensorrt_llm/`)

- **`llmapi/`** — High-level Python API (`LLM` class) for model loading and inference. Entry point for most users. Handles checkpoint loading, engine building, and generation.
- **`models/`** — Model architecture definitions (Llama, GPT, DeepSeek, Falcon, Qwen, etc.). Each model subdir contains checkpoint conversion and model definition. `automodel.py` provides automatic model selection.
- **`layers/`** — Neural network layer primitives (attention, linear, MLP, MoE, normalization, embedding, SSM).
- **`_torch/`** — PyTorch-native execution path (alternative to TensorRT engine path):
  - `models/` — PyTorch model implementations
  - `auto_deploy/` — Automatic model deployment/optimization
  - `compilation/` — Torch compilation and optimization passes
  - `pyexecutor/` — Python-based executor for the PyTorch path
  - `speculative/` — Speculative decoding implementations
- **`runtime/`** — TensorRT engine runtime: model runners, KV cache management, session management.
- **`executor/`** — Python bindings to the C++ executor for batched inference.
- **`serve/`** — OpenAI-compatible API server (`trtllm-serve`), including disaggregated serving support.
- **`quantization/`** — Quantization utilities (FP8, INT8, INT4, FP4).
- **`scaffolding/`** — Agentic/multi-step inference framework (controller, workers, tasks).
- **`plugin/`** — TensorRT plugin registration and management.

### C++ Runtime (`cpp/`)

- **`cpp/include/tensorrt_llm/`** — Public C++ headers organized by component: `batch_manager`, `executor`, `runtime`, `kernels`, `layers`, `plugins`.
- **`cpp/tensorrt_llm/`** — C++ implementation source.
- **`cpp/kernels/`** — Standalone CUDA kernels (FMHA, XQA, etc.).
- **`cpp/tests/`** — C++ Google Test-based tests.

The C++ executor (`cpp/include/tensorrt_llm/executor/`) is the core inference engine that manages batching, scheduling, and KV cache. Python binds to it via pybind11 (`tensorrt_llm/bindings`).

### Key Data Flow

1. **User** → `LLM` class (llmapi) → loads model from HuggingFace/checkpoint
2. **Model definition** (models/) → builds TensorRT engine or uses PyTorch path (_torch/)
3. **Executor** (C++ executor via bindings) → manages request batching, KV cache, scheduling
4. **Kernels** (C++ kernels) → optimized CUDA kernels for attention, GEMM, etc.

### Other Notable Directories

- **`examples/`** — Usage examples including model-specific scripts, disaggregated serving, auto_deploy.
- **`benchmarks/`** — Performance benchmarking tools.
- **`3rdparty/`** — Third-party dependencies (cutlass, etc.).

## Coding Conventions

### C++
- Allman brace style, 4-space indentation, 120-char line limit
- Naming: `FooBarClass` (types), `localFooBar` (variables/methods), `mMemberVar` (class members), `kCONSTANT_NAME` (constants), `gGlobalVar` (globals)
- Format with clang-format before submitting

### Python
- Line length: 80 chars (yapf/isort files), 100 chars (ruff files)
- Use `pre-commit` hooks to auto-format
