# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from transformers.utils.device_context import DeviceContext, device_context


def test_move_tensor_to_cpu():
    ctx = DeviceContext("cpu")
    tensor = torch.randn(2, 3, device="cpu")
    moved = ctx.move(tensor)
    assert moved.device.type == "cpu"


def test_nested_contexts_use_inner_device():
    outer = DeviceContext("cpu")
    with outer:
        inner = DeviceContext("cpu")
        with inner:
            assert inner.active_device.type == "cpu"


def test_move_dict_of_tensors():
    ctx = DeviceContext("cpu")
    batch = {"input_ids": torch.ones(2, 4, dtype=torch.long)}
    moved = ctx.move(batch)
    assert moved["input_ids"].device.type == "cpu"


def test_invalid_device_raises():
    with pytest.raises((RuntimeError, ValueError)):
        DeviceContext("not-a-real-device-name-xyz")


def test_device_context_helper():
    with device_context("cpu") as ctx:
        t = ctx.move(torch.zeros(1))
        assert t.device.type == "cpu"


def test_move_module_parameters():
    module = torch.nn.Linear(4, 2)
    ctx = DeviceContext("cpu")
    moved = ctx.move_module(module)
    assert next(moved.parameters()).device.type == "cpu"
