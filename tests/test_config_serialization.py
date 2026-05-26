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

"""Characterization tests for config serialization contracts (WO-027)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from transformers.models.auto.configuration_auto import CONFIG_MAPPING


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "config_serialization"
LEGACY_FIXTURE = FIXTURES_DIR / "legacy" / "bert_base_uncased_v4.json"

REPRESENTATIVE_MODEL_TYPES = [
    "bert",
    "gpt2",
    "roberta",
    "distilbert",
    "albert",
    "t5",
    "bart",
    "xlnet",
    "electra",
    "deberta",
    "bloom",
    "llama",
    "mistral",
    "qwen2",
    "gemma",
    "vit",
    "clip",
    "whisper",
    "wav2vec2",
    "gpt_neox",
]

REPRESENTATIVE_MODEL_TYPES = [m for m in REPRESENTATIVE_MODEL_TYPES if m in CONFIG_MAPPING]
assert len(REPRESENTATIVE_MODEL_TYPES) >= 20


def _normalize_json_value(value):
    if isinstance(value, dict):
        normalized = {}
        for key, item in value.items():
            if isinstance(key, str) and key.isdigit():
                key = int(key)
            normalized[key] = _normalize_json_value(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    return value


def _load_json(path: Path) -> dict:
    return _normalize_json_value(json.loads(path.read_text(encoding="utf-8")))


def _config_for(model_type: str):
    return CONFIG_MAPPING[model_type]()


@pytest.mark.parametrize("model_type", REPRESENTATIVE_MODEL_TYPES)
def test_to_dict_matches_golden(model_type: str):
    config = _config_for(model_type)
    golden = _load_json(FIXTURES_DIR / f"{model_type}_to_dict.json")
    assert config.to_dict() == golden


@pytest.mark.parametrize("model_type", REPRESENTATIVE_MODEL_TYPES)
def test_to_json_string_matches_golden(model_type: str):
    config = _config_for(model_type)
    golden = (FIXTURES_DIR / f"{model_type}_to_json_string.json").read_text(encoding="utf-8")
    assert config.to_json_string(use_diff=False) == golden


@pytest.mark.parametrize("model_type", REPRESENTATIVE_MODEL_TYPES)
def test_save_pretrained_round_trip(model_type: str):
    config = _config_for(model_type)
    config_cls = CONFIG_MAPPING[model_type]
    with tempfile.TemporaryDirectory() as tmp_dir:
        config.save_pretrained(tmp_dir)
        reloaded = config_cls.from_pretrained(tmp_dir)
    assert reloaded.to_dict() == config.to_dict()


def test_backward_compatible_legacy_bert_config():
    """Loading a v4-style snapshot must preserve serialized attribute values."""
    from transformers import BertConfig

    legacy = _load_json(LEGACY_FIXTURE)
    loaded = BertConfig.from_dict(legacy)
    for key, expected in legacy.items():
        if key == "transformers_version":
            continue
        assert getattr(loaded, key) == expected, f"mismatch on {key}"
