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

"""Common downstream import patterns (WO-005)."""

import pytest


@pytest.mark.star_import
def test_import_transformers_package():
    import transformers

    assert hasattr(transformers, "__version__")
    assert transformers.__version__


@pytest.mark.star_import
def test_multi_symbol_import_pattern():
    """Downstream packages often import several public symbols at once."""
    namespace: dict = {}
    exec(
        "from transformers import AutoConfig, AutoTokenizer, pipeline, Trainer",
        {},
        namespace,
    )
    assert "AutoConfig" in namespace
    assert "AutoTokenizer" in namespace
    assert callable(namespace["pipeline"])
    assert namespace["Trainer"].__name__ == "Trainer"


@pytest.mark.auto_classes
def test_from_transformers_import_auto_model():
    pytest.importorskip("torch")
    from transformers import AutoModel

    assert hasattr(AutoModel, "from_pretrained")


@pytest.mark.auto_classes
def test_from_transformers_import_auto_tokenizer():
    from transformers import AutoTokenizer

    assert hasattr(AutoTokenizer, "from_pretrained")


@pytest.mark.auto_classes
def test_from_transformers_import_auto_config():
    from transformers import AutoConfig

    assert hasattr(AutoConfig, "from_pretrained")


@pytest.mark.auto_classes
def test_from_transformers_import_pretrained_base():
    from transformers import PreTrainedModel, PreTrainedConfig

    assert PreTrainedModel.__name__ == "PreTrainedModel"
    assert PreTrainedConfig.__name__ == "PreTrainedConfig"


@pytest.mark.submodule_access
def test_models_bert_modeling_import():
    pytest.importorskip("torch")
    from transformers.models.bert.modeling_bert import BertModel

    assert BertModel.__name__ == "BertModel"


@pytest.mark.submodule_access
def test_models_bert_configuration_import():
    from transformers.models.bert.configuration_bert import BertConfig

    assert BertConfig.__name__ == "BertConfig"


@pytest.mark.submodule_access
def test_transformers_utils_import():
    from transformers.utils import is_torch_available

    assert callable(is_torch_available)


@pytest.mark.submodule_access
def test_transformers_models_auto_submodule():
    from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification

    assert hasattr(AutoModelForSequenceClassification, "from_pretrained")


@pytest.mark.submodule_access
def test_transformers_generation_config_import():
    from transformers import GenerationConfig

    assert GenerationConfig.__name__ == "GenerationConfig"


@pytest.mark.submodule_access
def test_transformers_logging_import():
    from transformers import logging as transformers_logging

    assert hasattr(transformers_logging, "get_logger")


@pytest.mark.trainer_api
def test_trainer_import():
    from transformers import Trainer, TrainingArguments

    assert Trainer.__name__ == "Trainer"
    assert TrainingArguments.__name__ == "TrainingArguments"


@pytest.mark.pipeline_api
def test_pipeline_import_from_root():
    from transformers import pipeline

    assert callable(pipeline)


@pytest.mark.pipeline_api
def test_pipeline_import_from_pipelines_submodule():
    from transformers.pipelines import pipeline as pipeline_fn

    assert callable(pipeline_fn)


@pytest.mark.submodule_access
def test_tokenization_utils_base_import():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    assert PreTrainedTokenizerBase.__name__ == "PreTrainedTokenizerBase"


@pytest.mark.submodule_access
def test_lazy_module_registers_public_api():
    import transformers

    assert "AutoModel" in transformers.__all__ or hasattr(transformers, "AutoModel")
