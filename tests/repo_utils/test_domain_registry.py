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

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES  # noqa: E402

REGISTRY_PATH = SRC / "transformers" / "models" / "domain_registry.json"
VALID_DOMAINS = {"nlp", "vision", "audio", "multimodal"}


def test_registry_covers_all_config_mapping_names():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert registry["model_count"] == len(CONFIG_MAPPING_NAMES)
    assert set(registry["models"]) == set(CONFIG_MAPPING_NAMES)


@pytest.mark.parametrize("model_type", list(CONFIG_MAPPING_NAMES.keys())[:20])
def test_sample_entries_have_valid_domains(model_type):
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    entry = registry["models"][model_type]
    assert entry["primary_domain"] in VALID_DOMAINS
    assert all(domain in VALID_DOMAINS for domain in entry["secondary_domains"])
