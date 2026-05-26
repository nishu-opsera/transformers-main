#!/usr/bin/env python3
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
"""Optional static check for protocol conformance stub (WO-012)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB = REPO_ROOT / "tests" / "fixtures" / "protocol_conformance_stub.py"
SRC_ROOT = REPO_ROOT / "src"


def main() -> int:
    if not STUB.exists():
        print(f"Missing conformance stub: {STUB}", file=sys.stderr)
        return 1

    ty = shutil.which("ty")
    if ty is None:
        print("ty not installed — skipping static protocol check (runtime tests still apply).")
        return 0

    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_ROOT) + (f"{subprocess.os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    result = subprocess.run([ty, "check", str(STUB)], cwd=REPO_ROOT, env=env, check=False)
    if result.returncode == 0:
        print("ty: protocol conformance stub passed.")
    else:
        print("ty: protocol conformance stub reported issues.", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
