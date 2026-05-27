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
UTILS = REPO_ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from track_modular_migration import build_progress  # noqa: E402


def test_wave1_migrated_files_use_modular_import():
    wave1 = REPO_ROOT / ".ci" / "modular_migration_wave1_files.txt"
    assert wave1.exists()
    paths = [line.strip() for line in wave1.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(paths) >= 26
    for rel in paths:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "modular_swin" in text or "SwinDropPath" in text


def test_progress_tracks_top_50():
    report = build_progress(50)
    assert report["summary"]["files_tracked"] == 50
