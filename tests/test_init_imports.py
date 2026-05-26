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

"""Characterization tests for root ``transformers`` import behavior (WO-026)."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest

from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers.utils.import_utils import _LazyModule


# Core public API symbols from the base ``_import_structure`` (non-model leaf modules).
CORE_PUBLIC_SYMBOLS = [
    "PreTrainedConfig",
    "PretrainedConfig",
    "GenerationConfig",
    "AutoConfig",
    "AutoTokenizer",
    "logging",
    "is_torch_available",
]

TORCH_PUBLIC_SYMBOLS = [
    "Trainer",
    "TrainingArguments",
    "pipeline",
    "AutoModel",
    "AutoProcessor",
    "AutoImageProcessor",
]

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

REPRESENTATIVE_MODEL_TYPES = [m for m in REPRESENTATIVE_MODEL_TYPES if m in CONFIG_MAPPING_NAMES]
assert len(REPRESENTATIVE_MODEL_TYPES) >= 20, "Need at least 20 mapped model types for characterization coverage"


@pytest.fixture
def transformers_module() -> ModuleType:
    return importlib.import_module("transformers")


@pytest.mark.parametrize("symbol", CORE_PUBLIC_SYMBOLS)
def test_core_public_symbol_importable(transformers_module, symbol):
    obj = getattr(transformers_module, symbol)
    assert obj is not None


@pytest.mark.parametrize("symbol", TORCH_PUBLIC_SYMBOLS)
def test_torch_public_symbol_importable(transformers_module, symbol):
    pytest.importorskip("torch")
    obj = getattr(transformers_module, symbol)
    assert obj is not None


def test_root_module_is_lazy(transformers_module):
    assert isinstance(transformers_module, _LazyModule)


def test_lazy_module_defers_model_submodule_until_access(transformers_module):
    pytest.importorskip("torch")
    modeling_bert = "transformers.models.bert.modeling_bert"
    if modeling_bert in sys.modules:
        del sys.modules[modeling_bert]
    assert modeling_bert not in sys.modules
    bert_model = getattr(transformers_module, "BertModel")
    assert bert_model is not None
    assert modeling_bert in sys.modules


def test_star_import_exposes_public_api():
    pytest.importorskip("torch")
    namespace: dict = {}
    exec("from transformers import *", {}, namespace)
    assert "PreTrainedConfig" in namespace
    assert "AutoConfig" in namespace


def test_submodule_model_import():
    pytest.importorskip("torch")
    from transformers.models.bert.modeling_bert import BertModel

    assert BertModel.__name__ == "BertModel"


@pytest.mark.parametrize("model_type", REPRESENTATIVE_MODEL_TYPES)
def test_auto_config_mapping_resolves(model_type):
    from transformers.configuration_utils import PreTrainedConfig

    config_cls = CONFIG_MAPPING[model_type]
    assert issubclass(config_cls, PreTrainedConfig)


@pytest.mark.parametrize("model_type", REPRESENTATIVE_MODEL_TYPES)
def test_auto_model_mapping_resolves(model_type):
    pytest.importorskip("torch")
    model_cls = MODEL_MAPPING[model_type]
    assert hasattr(model_cls, "from_pretrained")


def test_auto_tokenizer_class_exposes_from_pretrained():
    from transformers import AutoTokenizer

    assert hasattr(AutoTokenizer, "from_pretrained")


def test_auto_processor_class_exposes_from_pretrained():
    from transformers import AutoProcessor

    assert hasattr(AutoProcessor, "from_pretrained")


def test_auto_image_processor_class_exposes_from_pretrained():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("PIL")
    from transformers import AutoImageProcessor

    assert hasattr(AutoImageProcessor, "from_pretrained")


@pytest.mark.parametrize(
    "domain,symbol",
    [
        ("nlp", "AutoModelForCausalLM"),
        ("vision", "AutoModel"),
        ("audio", "AutoModel"),
        ("multimodal", "AutoModel"),
    ],
)
def test_domain_registry_opt_in_import(domain, symbol):
    module = importlib.import_module(f"transformers.domains.{domain}")
    assert hasattr(module, symbol)


def test_import_structure_symbols_listed_in_all(transformers_module):
    expected = CORE_PUBLIC_SYMBOLS + TORCH_PUBLIC_SYMBOLS
    missing = [name for name in expected if name not in transformers_module.__all__]
    assert not missing, f"Expected symbols missing from __all__: {missing}"
