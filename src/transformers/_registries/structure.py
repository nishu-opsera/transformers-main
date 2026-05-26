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
"""Import-structure helpers for domain sub-registries (WO-010, WO-011)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils.import_utils import define_import_structure

if TYPE_CHECKING:
    from ..utils.import_utils import IMPORT_STRUCTURE_T

_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models" / "domain_registry.json"
_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"
_SHARED_MODEL_PACKAGES = frozenset({"auto", "deprecated"})


def load_domain_registry() -> dict:
    return json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def model_folder_from_import_path(module_path: str) -> str | None:
    if not module_path.startswith("models."):
        return None
    parts = module_path.split(".")
    return parts[1] if len(parts) >= 2 else None


def merge_import_structures(*structures: IMPORT_STRUCTURE_T) -> IMPORT_STRUCTURE_T:
    """Deep-merge lazy import structures keyed by backend frozensets."""
    merged: IMPORT_STRUCTURE_T = {}
    for structure in structures:
        for backends, modules in structure.items():
            bucket = merged.setdefault(backends, {})
            for module_path, names in modules.items():
                bucket.setdefault(module_path, set()).update(names)
    return merged


def _registry_keys_for_folder(folder: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys((folder, folder.replace("_", "-"), folder.replace("-", "_"))))


def _primary_domain_for_folder(folder: str, registry: dict) -> str | None:
    if folder in _SHARED_MODEL_PACKAGES:
        return None
    models = registry["models"]
    for key in _registry_keys_for_folder(folder):
        entry = models.get(key)
        if entry is not None:
            return entry["primary_domain"]
    return None


def filter_models_import_structure(
    structure: IMPORT_STRUCTURE_T,
    *,
    domains: set[str],
    include_shared_packages: bool = False,
) -> IMPORT_STRUCTURE_T:
    registry = load_domain_registry()
    filtered: IMPORT_STRUCTURE_T = {}
    for backends, modules in structure.items():
        kept: dict[str, set[str]] = {}
        for module_path, names in modules.items():
            folder = model_folder_from_import_path(module_path)
            if folder is None:
                continue
            if include_shared_packages and folder in _SHARED_MODEL_PACKAGES:
                kept[module_path] = set(names)
                continue
            primary = _primary_domain_for_folder(folder, registry)
            if primary in domains:
                kept[module_path] = set(names)
        if kept:
            filtered[backends] = kept
    return filtered


@lru_cache
def full_models_import_structure() -> IMPORT_STRUCTURE_T:
    return define_import_structure(_MODELS_ROOT, prefix="models")


@lru_cache
def build_registry_import_structure(domain: str) -> IMPORT_STRUCTURE_T:
    """Import structure for a single primary domain plus shared ``models.auto``."""
    full = full_models_import_structure()
    domain_slice = filter_models_import_structure(full, domains={domain})
    auto_slice = filter_models_import_structure(full, domains=set(), include_shared_packages=True)
    return merge_import_structures(domain_slice, auto_slice)


def _unmapped_models_import_structure(
    full: IMPORT_STRUCTURE_T, mapped: IMPORT_STRUCTURE_T
) -> IMPORT_STRUCTURE_T:
    """Paths present in ``full`` but not yet assigned to a domain fragment."""
    mapped_paths: set[str] = set()
    for modules in mapped.values():
        mapped_paths.update(modules)
    unmapped: IMPORT_STRUCTURE_T = {}
    for backends, modules in full.items():
        orphan_modules: dict[str, set[str]] = {}
        for module_path, names in modules.items():
            if module_path not in mapped_paths:
                orphan_modules[module_path] = set(names)
        if orphan_modules:
            unmapped[backends] = orphan_modules
    return unmapped


_ALL_DOMAINS = frozenset({"nlp", "vision", "audio", "multimodal"})


@lru_cache
def build_composed_root_models_structure() -> IMPORT_STRUCTURE_T:
    """Models tree split across all four domain registries (WO-010, WO-011)."""
    full = full_models_import_structure()
    domain_fragments = [
        filter_models_import_structure(full, domains={domain}) for domain in sorted(_ALL_DOMAINS)
    ]
    auto_slice = filter_models_import_structure(full, domains=set(), include_shared_packages=True)
    partial = merge_import_structures(*domain_fragments, auto_slice)
    unmapped = _unmapped_models_import_structure(full, partial)
    return merge_import_structures(partial, unmapped)
