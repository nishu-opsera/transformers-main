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

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers.base_abstractions import (  # noqa: E402
    config_dict_for_save,
    model_type_from_config,
)
from transformers.protocols import (  # noqa: E402
    ConfigProtocol,
    ModelConfigConsumer,
    ModelWithConfigProtocol,
    PreTrainedConfigProtocol,
    assert_config_protocol,
    assert_model_config_consumer,
)


@dataclass
class _StubConfig:
    model_type: str = "bert"
    hidden_size: int = 768

    def to_dict(self):
        return {"model_type": self.model_type, "hidden_size": self.hidden_size}

    def to_json_string(self, use_diff: bool = True) -> str:
        return '{"model_type": "bert"}'


@dataclass
class _IncompleteConfig:
    """Missing serialization methods — must not satisfy ConfigProtocol."""

    model_type: str = "broken"


@dataclass
class _StubModel:
    config: _StubConfig

    def save_pretrained(self, save_directory: str, **kwargs):
        return None


def test_config_protocol_runtime_checkable():
    config = _StubConfig()
    assert isinstance(config, ConfigProtocol)
    assert isinstance(config, PreTrainedConfigProtocol)
    assert isinstance(config, ModelConfigConsumer)
    assert model_type_from_config(config) == "bert"
    assert config_dict_for_save(config)["hidden_size"] == 768
    assert_config_protocol(config)
    assert_model_config_consumer(config)


def test_incomplete_config_rejected():
    bad = _IncompleteConfig()
    assert not isinstance(bad, ConfigProtocol)
    with pytest.raises(TypeError):
        assert_config_protocol(bad)


def test_model_protocol_runtime_checkable():
    model = _StubModel(_StubConfig())
    assert isinstance(model, ModelWithConfigProtocol)


def test_bert_config_satisfies_config_protocol():
    from transformers import BertConfig

    config = BertConfig()
    assert isinstance(config, ConfigProtocol)
    assert isinstance(config, ModelConfigConsumer)
    assert config.model_type == "bert"


def test_protocol_typing_check_script():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "utils" / "check_protocol_typing.py")],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
