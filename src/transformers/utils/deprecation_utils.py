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

"""Standardized deprecation helpers for public API changes (WO-023)."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from .deprecation import deprecate_kwarg


_F = TypeVar("_F", bound=Callable[..., object])


def format_deprecation_message(
    name: str,
    *,
    replacement: str | None,
    version: str,
    migration_guide_url: str | None = None,
) -> str:
    """Build the canonical deprecation warning message."""
    if replacement:
        message = (
            f"{name} is deprecated and will be removed in version {version}. "
            f"Use {replacement} instead."
        )
    else:
        message = f"{name} is deprecated and will be removed in version {version}."
    if migration_guide_url:
        message += f" See {migration_guide_url} for migration guidance."
    return message


def _warn_deprecation(
    name: str,
    *,
    replacement: str | None,
    version: str,
    migration_guide_url: str | None = None,
    stacklevel: int = 3,
) -> None:
    warnings.warn(
        format_deprecation_message(
            name,
            replacement=replacement,
            version=version,
            migration_guide_url=migration_guide_url,
        ),
        FutureWarning,
        stacklevel=stacklevel,
    )


def deprecated_function(
    name: str,
    *,
    replacement: str,
    version: str,
    migration_guide_url: str | None = None,
) -> Callable[[_F], _F]:
    """Decorator marking a public function as deprecated."""

    def decorator(func: _F) -> _F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _warn_deprecation(
                name,
                replacement=replacement,
                version=version,
                migration_guide_url=migration_guide_url,
            )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_class(
    name: str,
    *,
    replacement: str,
    version: str,
    migration_guide_url: str | None = None,
) -> Callable[[type], type]:
    """Class decorator emitting a deprecation warning when the class is instantiated."""

    def decorator(cls: type) -> type:
        original_init = cls.__init__

        @wraps(original_init)
        def patched_init(self, *args, **kwargs):
            _warn_deprecation(
                name,
                replacement=replacement,
                version=version,
                migration_guide_url=migration_guide_url,
                stacklevel=3,
            )
            return original_init(self, *args, **kwargs)

        cls.__init__ = patched_init  # type: ignore[method-assign]
        return cls

    return decorator


def deprecated_argument(
    old_name: str,
    *,
    new_name: str | None = None,
    version: str,
    migration_guide_url: str | None = None,
    **deprecate_kwarg_kwargs,
) -> Callable[[_F], _F]:
    """Decorator for deprecated keyword arguments; wraps ``deprecate_kwarg`` with unified messaging."""
    additional_message = format_deprecation_message(
        old_name,
        replacement=new_name or old_name,
        version=version,
        migration_guide_url=migration_guide_url,
    )
    return deprecate_kwarg(
        old_name,
        version=version,
        new_name=new_name,
        additional_message=additional_message,
        **deprecate_kwarg_kwargs,
    )


def iter_registered_deprecations() -> list[dict[str, str | None]]:
    """Return metadata for deprecations declared via ``deprecated_*`` decorators in ``src/transformers``."""
    from pathlib import Path

    import ast

    root = Path(__file__).resolve().parents[1]
    records: list[dict[str, str | None]] = []

    for path in sorted(root.rglob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name):
                continue
            if func.id not in {"deprecated_function", "deprecated_class", "deprecated_argument", "deprecate_kwarg"}:
                continue
            record: dict[str, str | None] = {
                "file": str(path.relative_to(root.parent)),
                "decorator": func.id,
                "name": None,
                "replacement": None,
                "version": None,
                "migration_guide_url": None,
            }
            for keyword in node.keywords:
                if keyword.arg in record and isinstance(keyword.value, ast.Constant):
                    record[keyword.arg] = str(keyword.value.value)
            if func.id == "deprecate_kwarg" and node.args:
                if isinstance(node.args[0], ast.Constant):
                    record["name"] = str(node.args[0].value)
            records.append(record)
    return records
