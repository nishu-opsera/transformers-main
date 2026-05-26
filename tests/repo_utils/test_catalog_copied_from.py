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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.catalog_copied_from import scan_file  # noqa: E402

CATALOG_PATH = REPO_ROOT / ".ci" / "copied_from_catalog.json"
# ForgeScore / WO-015 estimates (~1,473 / ~326); regenerated catalog as of WO-015 commit.
BASELINE_ANNOTATIONS = 1430
BASELINE_FILES = 309
TOLERANCE = 0.05


def test_scan_file_parses_class_and_method(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        '''
class Foo:
    # Copied from transformers.models.bert.modeling_bert.BertLayer with changes
    def bar(self):
        pass
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    entries = scan_file(sample)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.target_name == "bar"
    assert entry.annotation_type == "method"
    assert entry.has_modifications is True
    assert entry.migration_complexity == "complex"


def test_catalog_on_disk_within_baseline_tolerance():
    assert CATALOG_PATH.exists(), "Run: python utils/catalog_copied_from.py"
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    total = catalog["summary"]["total_annotations"]
    files = catalog["summary"]["total_files"]
    assert abs(total - BASELINE_ANNOTATIONS) / BASELINE_ANNOTATIONS <= TOLERANCE
    assert abs(files - BASELINE_FILES) / BASELINE_FILES <= TOLERANCE
    assert catalog["summary"]["priority_migration_files"]
    assert catalog["summary"]["annotation_types"]
