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

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "copied_from_diff"
SCRIPT = REPO_ROOT / "utils" / "check_copied_from_regression.py"


def _run_check(diff_name: str, *, block: str | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if block is not None:
        env["COPIED_FROM_BLOCK_NEW"] = block
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--diff-file", str(FIXTURES / diff_name)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_warns_on_new_annotation_in_diff():
    result = _run_check("new_annotation.diff", block="0")
    assert result.returncode == 0
    assert "WARNING" in result.stderr
    assert "modular_transformers" in result.stderr
    assert "modeling_example.py" in result.stderr


def test_blocks_when_flag_enabled():
    result = _run_check("new_annotation.diff", block="1")
    assert result.returncode == 1
    assert "WARNING" in result.stderr


def test_clean_diff_exits_zero():
    result = _run_check("no_new_annotation.diff", block="1")
    assert result.returncode == 0
    assert "No new" in result.stdout


def test_parse_added_lines_directly():
    sys.path.insert(0, str(REPO_ROOT))
    from utils.check_copied_from_regression import parse_added_copied_from_lines  # noqa: E402

    diff_text = (FIXTURES / "new_annotation.diff").read_text(encoding="utf-8")
    findings = parse_added_copied_from_lines(diff_text)
    assert len(findings) == 1
    assert findings[0].file_path.endswith("modeling_example.py")
