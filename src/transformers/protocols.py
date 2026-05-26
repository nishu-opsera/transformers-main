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
"""Shared typing protocols for config/model boundaries (WO-007).

Neutral contracts imported by both configuration and modeling layers without
creating mutual runtime dependencies between those modules.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PreTrainedConfigProtocol(Protocol):
    """Structural contract for configuration objects used by modeling code."""

    model_type: str

    def to_dict(self) -> dict[str, Any]:
        ...

    def to_json_string(self, use_diff: bool = True) -> str:
        ...


@runtime_checkable
class ModelWithConfigProtocol(Protocol):
    """Structural contract for models referencing a configuration object."""

    config: PreTrainedConfigProtocol

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        ...
