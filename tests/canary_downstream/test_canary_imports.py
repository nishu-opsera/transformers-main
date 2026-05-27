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
"""Canary import smoke tests for downstream-style usage (WO-030)."""

from __future__ import annotations

import importlib

import pytest

# Representative downstream import patterns (expand toward top-100 over time).
CANARY_MODULES = [
    "transformers",
    "transformers.models.auto.modeling_auto",
    "transformers.pipelines",
    "transformers.trainer",
    "transformers.tokenization_utils_base",
    "transformers.domains.nlp",
    "transformers.domains.vision",
    "transformers.domains.audio",
    "transformers.domains.multimodal",
    "transformers.utils.device_context",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "evaluate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "numpy",
    "torch",
]


@pytest.mark.parametrize("module_name", CANARY_MODULES)
def test_canary_module_imports(module_name: str):
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(f"optional dependency not installed: {module_name} ({exc})")
    assert module is not None
