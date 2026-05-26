# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Multimodal domain lazy registry (alias for ``transformers._registries.multimodal``)."""

import sys

from .._registries import multimodal as _multimodal_registry

sys.modules[__name__] = sys.modules[_multimodal_registry.__name__]
