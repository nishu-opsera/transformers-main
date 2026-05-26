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

import warnings

import pytest

from transformers.utils.deprecation_utils import (
    deprecated_argument,
    deprecated_class,
    deprecated_function,
    format_deprecation_message,
)


def test_format_deprecation_message_with_url():
    message = format_deprecation_message(
        "legacy_fn",
        replacement="modern_fn",
        version="6.0.0",
        migration_guide_url="https://example.com/migrate",
    )
    assert "legacy_fn is deprecated" in message
    assert "6.0.0" in message
    assert "modern_fn" in message
    assert "https://example.com/migrate" in message


def test_deprecated_function_emits_warning():
    @deprecated_function("legacy_fn", replacement="modern_fn", version="6.0.0")
    def legacy_fn():
        return 1

    with pytest.warns(FutureWarning, match="legacy_fn is deprecated"):
        assert legacy_fn() == 1


def test_deprecated_class_emits_warning():
    @deprecated_class("LegacyClass", replacement="ModernClass", version="6.0.0")
    class LegacyClass:
        def __init__(self):
            self.value = 42

    with pytest.warns(FutureWarning, match="LegacyClass is deprecated"):
        obj = LegacyClass()
    assert obj.value == 42


def test_deprecated_argument_integrates_with_deprecate_kwarg():
    @deprecated_argument("old_flag", new_name="new_flag", version="6.0.0")
    def sample(*, new_flag=False):
        return new_flag

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert sample(old_flag=True) is True
    assert any("old_flag" in str(w.message) for w in caught)
