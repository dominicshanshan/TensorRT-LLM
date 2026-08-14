# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Piecewise CUDA graph (PCG) correctness under the Ray executor.

Three gaps previously prevented PCG from working with orchestrator_type="ray":

1. build_custom_passes used mpi_world_size() which returns 1 per Ray actor,
   making the multi-GPU AR-fusion branch dead.
2. AllReduce.forward accessed mapping.tp_group_pg / mapping.tp_group at
   Dynamo trace time, hitting @torch.compiler.disable'd _get_mesh_dim_by_name.
3. _init_userbuffers called the MPI-only UB C++ bootstrap, crashing when
   enable_userbuffers=True was set under Ray.

test_build_custom_passes_ar_fusions_registered covers Fix 1 in isolation.
test_pcg_ray_correctness covers Fixes 1+2+3 end-to-end.
"""

import pytest
import torch

from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams, TorchCompileConfig
from tensorrt_llm.mapping import Mapping

# ---------------------------------------------------------------------------
# Fix 1 unit test — no GPU required, pure Python / pattern-matcher logic.
# ---------------------------------------------------------------------------


def test_build_custom_passes_ar_fusions_registered():
    """AR fusion patterns must be registered for tp_size>1 under Ray.

    Before Fix 1, mpi_world_size()==1 per Ray actor kept the multi-GPU
    branch dead so no allreduce+residual+RMSNorm fusion patterns were
    installed, and no DISABLE_LAMPORT_REDUCE_NORM_FUSION guard was set.
    """
    from tensorrt_llm._torch.compilation.backend import Backend

    mapping = Mapping(world_size=2, tp_size=2, rank=0)
    passes = Backend.build_custom_passes(enable_userbuffers=False,
                                         mapping=mapping)
    pass_names = [p.pass_name for p in passes]

    assert any("ar_residual_norm" in name for name in pass_names), (
        "AR fusion pass not registered for tp_size>1 under Ray. "
        f"Got pass names: {pass_names}")
    assert not any(name == "add_norm_quant" for name in pass_names), (
        "add_norm_quant (single-GPU path) present — multi-GPU branch was "
        "not entered. Got pass names: {pass_names}")


# ---------------------------------------------------------------------------
# End-to-end integration tests — require 2 GPUs.
# ---------------------------------------------------------------------------

_MODEL = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
_PROMPTS = [
    "The capital of France is",
    "PyTorch is a popular deep learning",
    "TensorRT-LLM accelerates",
]
_SAMPLING = SamplingParams(max_tokens=32, temperature=0.0, top_p=1.0)

# Reduced capture list for fast warmup in a test context.
_CAPTURE_NUM_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _make_llm(torch_compile_config=None, allreduce_strategy="NCCL"):
    return LLM(
        model=str(llm_models_root() / _MODEL),
        backend="pytorch",
        orchestrator_type="ray",
        tensor_parallel_size=2,
        max_batch_size=8,
        max_num_tokens=512,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.6),
        allreduce_strategy=allreduce_strategy,
        torch_compile_config=torch_compile_config,
    )


def _generate(llm):
    outputs = llm.generate(_PROMPTS, _SAMPLING)
    return [o.outputs[0].token_ids for o in outputs]


@pytest.mark.gpu2
def test_pcg_ray_correctness():
    """PCG under Ray must produce token-identical output to the eager baseline.

    Validates Fix 1 (AR fusion patterns registered) and Fix 2 (fullgraph
    traces without hitting @compiler.disable'd properties).  Both LLM instances
    use allreduce_strategy=NCCL to keep the allreduce kernel identical between
    eager and compiled paths, ensuring a clean token-level comparison.
    """
    # Eager baseline — no torch.compile, plain NCCL allreduce.
    llm_base = _make_llm(torch_compile_config=None, allreduce_strategy="NCCL")
    try:
        base_tokens = _generate(llm_base)
    finally:
        del llm_base
        torch.cuda.empty_cache()

    # PCG path — enable_fullgraph + piecewise CUDA graph, UB disabled.
    llm_pcg = _make_llm(
        torch_compile_config=TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            enable_userbuffers=False,
            capture_num_tokens=_CAPTURE_NUM_TOKENS,
        ),
        allreduce_strategy="NCCL",
    )
    try:
        pcg_tokens = _generate(llm_pcg)
    finally:
        del llm_pcg
        torch.cuda.empty_cache()

    assert base_tokens == pcg_tokens, (
        "PCG output differs from eager baseline.\n"
        f"Baseline : {base_tokens}\n"
        f"PCG      : {pcg_tokens}")


@pytest.mark.gpu2
def test_pcg_ray_ub_true_no_crash():
    """enable_userbuffers=True must not crash under Ray (Fix 3).

    The UB C++ bootstrap (MPI-only) is skipped by the mpi_disabled() guard
    in _init_userbuffers; the LLM degrades gracefully to UB=false and should
    produce the same tokens as the explicit UB=false run.
    """
    # Reference: PCG with explicit UB=false.
    llm_ref = _make_llm(
        torch_compile_config=TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            enable_userbuffers=False,
            capture_num_tokens=_CAPTURE_NUM_TOKENS,
        ),
        allreduce_strategy="NCCL",
    )
    try:
        ref_tokens = _generate(llm_ref)
    finally:
        del llm_ref
        torch.cuda.empty_cache()

    # UB=True — should degrade silently and produce identical output.
    llm_ub = _make_llm(
        torch_compile_config=TorchCompileConfig(
            enable_fullgraph=True,
            enable_piecewise_cuda_graph=True,
            enable_userbuffers=True,  # Fix 3: must not crash
            capture_num_tokens=_CAPTURE_NUM_TOKENS,
        ),
        allreduce_strategy="NCCL",
    )
    try:
        ub_tokens = _generate(llm_ub)
    finally:
        del llm_ub
        torch.cuda.empty_cache()

    assert ref_tokens == ub_tokens, (
        "PCG+UB=True output differs from PCG+UB=False; "
        "graceful degradation should give identical results.\n"
        f"UB=False: {ref_tokens}\n"
        f"UB=True : {ub_tokens}")
