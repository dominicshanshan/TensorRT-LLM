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
"""The KV cache capacity estimation executor must not register its KV pools with the transfer agent.

Its pools are torn down right after the estimation; registering them with NIXL can leave their device
memory pinned after deregistration, and the final KV cache allocation then fails. The serving executor
keeps registering its own pools.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.disaggregation import kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.native import transfer as transfer_module
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig

pytestmark = pytest.mark.cpu_only


def test_transfer_worker_config_registers_kv_cache_by_default() -> None:
    config = transfer_module.TransferWorkerConfig(
        kv_cache_manager=Mock(), device_id=0, instance_name="instance"
    )

    assert config.register_kv_cache is True


@pytest.mark.parametrize("register_kv_cache", [True, False])
def test_transfer_worker_registers_kv_cache_only_when_requested(
    register_kv_cache: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_setup_transfer_engine skips only the KV pool registration; the aux buffer, sender and
    receiver are still set up."""
    monkeypatch.setattr(transfer_module.torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(transfer_module.cudart, "cudaSetDevice", lambda device: (0,))
    monkeypatch.setattr(transfer_module, "CUASSERT", lambda result: None)
    agent = Mock(name="agent")
    agent.get_local_agent_desc.return_value = b"agent-desc"
    monkeypatch.setattr(transfer_module, "_create_nixl_agent", Mock(return_value=agent))
    monkeypatch.setattr(transfer_module, "Sender", Mock(name="Sender"))
    monkeypatch.setattr(transfer_module, "Receiver", Mock(name="Receiver"))

    worker = object.__new__(transfer_module.TransferWorker)
    worker._config = transfer_module.TransferWorkerConfig(
        kv_cache_manager=SimpleNamespace(mapping=SimpleNamespace(rank=0, world_size=1)),
        device_id=0,
        instance_name="instance",
        register_kv_cache=register_kv_cache,
    )
    worker._rank_info = SimpleNamespace(instance_name="instance", instance_rank=0)
    worker._aux_buffer = object()
    worker._peer_registrar = Mock()
    worker._retirement_watchdog = None
    register_kv = Mock(name="_register_kv_cache")
    register_aux = Mock(name="_register_aux_buffer")
    worker._register_kv_cache = register_kv
    worker._register_aux_buffer = register_aux

    worker._setup_transfer_engine()

    assert register_kv.call_count == (1 if register_kv_cache else 0)
    register_aux.assert_called_once_with()
    assert worker._sender is transfer_module.Sender.return_value
    assert worker._receiver is transfer_module.Receiver.return_value
    assert worker._rank_info.transfer_engine_info == b"agent-desc"


@pytest.mark.parametrize("register_kv_cache", [True, False])
def test_python_transceiver_factory_forwards_register_kv_cache(
    register_kv_cache: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = object.__new__(KVCacheManagerV2)
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime="PYTHON")
    constructor = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.disaggregation.transceiver.KvCacheTransceiverV2", constructor
    )

    result = transceiver_module.create_kv_cache_transceiver(
        None, None, manager, None, config, register_kv_cache=register_kv_cache
    )

    assert result is constructor.return_value
    constructor.assert_called_once_with(
        None, None, manager, config, register_kv_cache=register_kv_cache
    )
