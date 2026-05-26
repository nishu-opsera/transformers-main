"""Configuration file that imports and uses modeling-layer symbols."""

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_utils import PreTrainedModel


class ViolatingConfig(PreTrainedConfig):
    model_type = "violating"

    def touch_model_layer(self):
        return PreTrainedModel
