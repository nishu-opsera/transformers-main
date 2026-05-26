# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Vision domain lazy registry (alias for ``transformers._registries.vision``)."""

import sys

from .._registries import vision as _vision_registry

sys.modules[__name__] = sys.modules[_vision_registry.__name__]
