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

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers.base_abstractions import (  # noqa: E402
    config_dict_for_save,
    model_type_from_config,
)
from transformers.protocols import (  # noqa: E402
    ModelWithConfigProtocol,
    PreTrainedConfigProtocol,
)


@dataclass
class _StubConfig:
    model_type: str = "bert"

    def to_dict(self):
        return {"model_type": self.model_type}

    def to_json_string(self, use_diff: bool = True) -> str:
        return '{"model_type": "bert"}'


@dataclass
class _StubModel:
    config: _StubConfig

    def save_pretrained(self, save_directory: str, **kwargs):
        return None


def test_config_protocol_runtime_checkable():
    config = _StubConfig()
    assert isinstance(config, PreTrainedConfigProtocol)
    assert model_type_from_config(config) == "bert"
    assert config_dict_for_save(config)["model_type"] == "bert"


def test_model_protocol_runtime_checkable():
    model = _StubModel(_StubConfig())
    assert isinstance(model, ModelWithConfigProtocol)
