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
"""Shared helpers for breaking configuration/model import cycles (WO-007)."""

from __future__ import annotations

from typing import Any

from .protocols import ConfigProtocol, ModelWithConfigProtocol


def model_type_from_config(config: ConfigProtocol) -> str:
    """Return the model type string from any config implementing the shared protocol."""
    return config.model_type


def assert_model_config_pair(model: ModelWithConfigProtocol) -> tuple[str, str]:
    """Validate that a model/config pair exposes consistent model_type identifiers."""
    model_type = model_type_from_config(model.config)
    if model_type != getattr(model, "config", None).model_type:
        raise ValueError("Model and config model_type mismatch.")
    return model_type, model_type


def config_dict_for_save(config: ConfigProtocol) -> dict[str, Any]:
    """Serialize config through the protocol surface (no modeling imports)."""
    return config.to_dict()
