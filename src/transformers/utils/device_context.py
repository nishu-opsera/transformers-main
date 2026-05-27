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
"""Centralized device placement for model boundaries (WO-018)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch


def _normalize_device(device: str | torch.device | int) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device(f"cuda:{device}" if device >= 0 else "cpu")
    return torch.device(device)


def _move_value(value: Any, device: torch.device, dtype: torch.dtype | None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_value(item, device, dtype) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_value(item, device, dtype) for item in value]
        return type(value)(moved)
    return value


class DeviceContext:
    """Context manager that pins tensor device/dtype for nested model forward passes.

    New models should enter ``DeviceContext`` at public forward boundaries instead of
    scattering ``.to(device)`` calls through submodules. Existing models may adopt
    incrementally (see ADR 002).
    """

    def __init__(
        self,
        device: str | torch.device | int,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> None:
        self.device = _normalize_device(device)
        self.dtype = dtype
        self.non_blocking = non_blocking
        self._stack: list[torch.device] = []

    def __enter__(self) -> DeviceContext:
        self._stack.append(self.device)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._stack:
            self._stack.pop()

    @property
    def active_device(self) -> torch.device:
        return self._stack[-1] if self._stack else self.device

    def move(self, value: Any) -> Any:
        return _move_value(value, self.active_device, self.dtype)

    def move_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if self.dtype is not None:
            return module.to(device=self.active_device, dtype=self.dtype)
        return module.to(device=self.active_device)


@contextmanager
def device_context(
    device: str | torch.device | int,
    *,
    dtype: torch.dtype | None = None,
) -> Iterator[DeviceContext]:
    ctx = DeviceContext(device, dtype=dtype)
    with ctx:
        yield ctx
