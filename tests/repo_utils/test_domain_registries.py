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
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from transformers._registries.structure import (  # noqa: E402
    build_composed_root_models_structure,
    build_registry_import_structure,
    filter_models_import_structure,
    full_models_import_structure,
    merge_import_structures,
    model_folder_from_import_path,
)
from transformers.models.auto.auto_mappings import CONFIG_MAPPING_NAMES  # noqa: E402


def _all_model_paths(structure) -> set[str]:
    paths: set[str] = set()
    for modules in structure.values():
        paths.update(modules)
    return paths


def test_model_folder_from_import_path():
    assert model_folder_from_import_path("models.bert.configuration_bert") == "bert"
    assert model_folder_from_import_path("generation.configuration_utils") is None


def test_nlp_registry_includes_bert_and_auto():
    nlp = build_registry_import_structure("nlp")
    paths = _all_model_paths(nlp)
    assert any(path.startswith("models.bert.") for path in paths)
    assert any(path.startswith("models.auto.") for path in paths)


def test_vision_registry_is_subset_of_full():
    full = full_models_import_structure()
    vision = build_registry_import_structure("vision")
    full_paths = _all_model_paths(full)
    vision_paths = _all_model_paths(vision)
    assert vision_paths <= full_paths


def test_composed_root_covers_all_model_packages():
    registry = __import__(
        "transformers._registries.structure", fromlist=["load_domain_registry"]
    ).load_domain_registry()
    full = full_models_import_structure()
    composed = build_composed_root_models_structure()
    full_folders = {model_folder_from_import_path(path) for path in _all_model_paths(full)}
    composed_folders = {model_folder_from_import_path(path) for path in _all_model_paths(composed)}
    full_folders.discard(None)
    composed_folders.discard(None)
    assert composed_folders == full_folders


@pytest.mark.parametrize(
    "domain,model_type",
    [
        ("nlp", "bert"),
        ("nlp", "gpt2"),
    ],
)
def test_domain_filter_keeps_primary_domain_models(domain, model_type):
    full = full_models_import_structure()
    filtered = filter_models_import_structure(full, domains={domain})
    paths = _all_model_paths(filtered)
    assert any(path.startswith(f"models.{model_type}.") for path in paths)


def test_merge_import_structures_unions_symbols():
    left = {frozenset(): {"models.bert.configuration_bert": {"BertConfig"}}}
    right = {frozenset(): {"models.bert.configuration_bert": {"BertModel"}, "models.auto.modeling_auto": {"AutoModel"}}}
    merged = merge_import_structures(left, right)
    assert merged[frozenset()]["models.bert.configuration_bert"] == {"BertConfig", "BertModel"}
    assert "AutoModel" in merged[frozenset()]["models.auto.modeling_auto"]
