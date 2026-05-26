"""Configuration file with no modeling-layer dependencies."""

from transformers.configuration_utils import PreTrainedConfig


class CleanConfig(PreTrainedConfig):
    model_type = "clean"

    def __init__(self, hidden_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
