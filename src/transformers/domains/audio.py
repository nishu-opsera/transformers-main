# Copyright 2026 The HuggingFace Team. All rights reserved.
"""Audio domain lazy registry (alias for ``transformers._registries.audio``)."""

import sys

from .._registries import audio as _audio_registry

sys.modules[__name__] = sys.modules[_audio_registry.__name__]
