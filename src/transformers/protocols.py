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
"""Config/model boundary protocols (WO-007, WO-012).

Boundary rules
--------------

**Configuration layer (``ConfigProtocol``)** — inert data only:

- May expose hyperparameters as attributes and serialize via ``to_dict`` / ``to_json_string``.
- Must **not** import or call modeling-layer functions (``PreTrainedModel``, ``torch.nn``, etc.).
- Must **not** delegate ``__getattr__`` / ``__call__`` into model code paths.

**Modeling layer (``ModelConfigConsumer``)** — may only depend on the read surface below:

- Read ``model_type`` and serialization helpers from configs passed into ``__init__``.
- Must not require config types that pull in ``modeling_*`` modules at import time.

These protocols are ``typing.Protocol`` contracts for static checkers (``ty``, Pyright, mypy)
and ``@runtime_checkable`` checks in tests. They do not alter runtime behavior of existing configs.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ConfigProtocol",
    "ModelConfigConsumer",
    "ModelWithConfigProtocol",
    "PreTrainedConfigProtocol",
    "assert_config_protocol",
    "assert_model_config_consumer",
]


@runtime_checkable
class ConfigProtocol(Protocol):
    """Full configuration contract: hyperparameter attributes plus serialization."""

    model_type: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration mapping."""
        ...

    def to_json_string(self, use_diff: bool = True) -> str:
        """Return a JSON string representation of this configuration."""
        ...


# WO-007 alias retained for importers that already reference this name.
PreTrainedConfigProtocol = ConfigProtocol


@runtime_checkable
class ModelConfigConsumer(Protocol):
    """Allowed read-only config surface for modeling-layer code (WO-012)."""

    model_type: str

    def to_dict(self) -> dict[str, Any]:
        ...


@runtime_checkable
class ModelWithConfigProtocol(Protocol):
    """Structural contract for models referencing a configuration object."""

    config: ConfigProtocol

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        ...


def assert_config_protocol(config: object) -> ConfigProtocol:
    """Runtime validation helper for tests and defensive checks."""
    if not isinstance(config, ConfigProtocol):
        raise TypeError(
            "Object does not satisfy ConfigProtocol (requires model_type, to_dict, to_json_string)."
        )
    return config


def assert_model_config_consumer(config: object) -> ModelConfigConsumer:
    """Runtime validation for the modeling-layer read surface."""
    if not isinstance(config, ModelConfigConsumer):
        raise TypeError("Object does not satisfy ModelConfigConsumer (requires model_type, to_dict).")
    return config
