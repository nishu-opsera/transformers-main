# Copyright 2022 Meta Platforms and The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Data2VecVision model."""

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...activations import ACT2FN
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    SemanticSegmenterOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import compile_compatible_method_lru_cache
from ...utils import auto_docstring, logging, torch_int
from .configuration_data2vec_vision import Data2VecVisionConfig


logger = logging.get_logger(__name__)


@auto_docstring(
    custom_intro="""
    Class for outputs of [`Data2VecVisionModel`].
    """
)
@dataclass
# Copied from transformers.models.beit.modeling_beit.BeitModelOutputWithPooling with Beit->Data2VecVision
class Data2VecVisionModelOutputWithPooling(BaseModelOutputWithPooling):
    r"""
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        Average of the last layer hidden states of the patch tokens (excluding the *[CLS]* token) if
        *config.use_mean_pooling* is set to True. If set to False, then the final hidden state of the *[CLS]* token
        will be returned.
    """


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitEmbeddings with Beit->Data2VecVision
class Data2VecVisionEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = Data2VecVisionPatchEmbeddings(config)
        self.patch_size = config.patch_size
        self.image_size = (
            config.image_size
            if isinstance(config.image_size, collections.abc.Iterable)
            else (config.image_size, config.image_size)
        )
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # Copied from transformers.models.vit.modeling_vit.ViTEmbeddings.interpolate_pos_encoding
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        embeddings, (patch_height, patch_width) = self.patch_embeddings(pixel_values)
        batch_size, seq_len, _ = embeddings.size()

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if self.position_embeddings is not None:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)

        embeddings = self.dropout(embeddings)

        return embeddings, (patch_height, patch_width)


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitPatchEmbeddings with Beit->Data2VecVision
class Data2VecVisionPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embeddings = self.projection(pixel_values.to(self.projection.weight.dtype))
        patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
        embeddings = embeddings.flatten(2).transpose(1, 2)

        return embeddings, (patch_height, patch_width)


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitSelfAttention with Beit->Data2VecVision
class Data2VecVisionSelfAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple | None = None) -> None:
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.has_relative_position_bias = bool(window_size)
        if self.has_relative_position_bias:
            self.relative_position_bias = Data2VecVisionRelativePositionBias(config, window_size=window_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        relative_position_bias: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        resolution: tuple[int] | None = None,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.has_relative_position_bias:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attention_scores = attention_scores + self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            attention_scores = attention_scores + relative_position_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitSdpaSelfAttention with Beit->Data2VecVision
class Data2VecVisionSdpaSelfAttention(Data2VecVisionSelfAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        relative_position_bias: torch.Tensor | None = None,
        interpolate_pos_encoding: bool = False,
        resolution: tuple[int] | None = None,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if output_attentions:
            logger.warning_once(
                f"{self.__class__.__name__} does not support `output_attentions=True`. The returned attention weights will "
                "be `None`. If you want to get attention weights, please set `attn_implementation='eager'` when loading the model."
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        query_layer = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        key_layer = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        value_layer = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        attn_bias = None
        if self.has_relative_position_bias:
            height, width = resolution
            window_size = (height // self.config.patch_size, width // self.config.patch_size)
            attn_bias = self.relative_position_bias(
                window_size, interpolate_pos_encoding, dim_size=hidden_states.shape[1]
            )

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            if attn_bias is None:
                attn_bias = relative_position_bias
            else:
                attn_bias += relative_position_bias

        scaling = 1 / math.sqrt(self.attention_head_size)
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attn_bias,
            dropout_p=self.config.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=scaling,
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, None


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitSelfOutput with Beit->Data2VecVision
class Data2VecVisionSelfOutput(nn.Module):
    """
    The residual connection is defined in Data2VecVisionLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, gamma=None) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


DATA2VEC_VISION_SELF_ATTENTION_CLASSES = {
    "eager": Data2VecVisionSelfAttention,
    "sdpa": Data2VecVisionSdpaSelfAttention,
}


# Copied from tests.models.beit.modeling_beit.BeitAttention with Beit->Data2VecVision, BEIT->DATA2VEC_VISION
class Data2VecVisionAttention(nn.Module):
    def __init__(self, config: Data2VecVisionConfig, window_size: tuple | None = None) -> None:
        super().__init__()
        self.attention = DATA2VEC_VISION_SELF_ATTENTION_CLASSES[config._attn_implementation](
            config, window_size=window_size
        )
        self.output = Data2VecVisionSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        relative_position_bias: Optional["Data2VecVisionRelativePositionBias"] = None,
        interpolate_pos_encoding: bool = False,
        resolution: tuple[int] | None = None,
    ) -> tuple[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        self_outputs = self.attention(
            hidden_states, output_attentions, relative_position_bias, interpolate_pos_encoding, resolution
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitIntermediate with Beit->Data2VecVision
class Data2VecVisionIntermediate(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitOutput with Beit->Data2VecVision
class Data2VecVisionOutput(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


from ..swin.modular_swin import SwinDropPath as Data2VecVisionDropPath



@auto_docstring
# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitPreTrainedModel with Beit->Data2VecVision,beit->data2vec_vision
class Data2VecVisionPreTrainedModel(PreTrainedModel):
    config: Data2VecVisionConfig
    base_model_prefix = "data2vec_vision"
    input_modalities = ("image",)
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Data2VecVisionLayer"]
    _keys_to_ignore_on_load_unexpected = [r".*relative_position_index.*"]
    _supports_sdpa = True

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, Data2VecVisionEmbeddings):
            init.zeros_(module.cls_token)
            if module.mask_token is not None:
                init.zeros_(module.mask_token)
            if module.position_embeddings is not None:
                init.zeros_(module.position_embeddings)
        elif isinstance(module, Data2VecVisionRelativePositionBias):
            init.zeros_(module.relative_position_bias_table)
        elif isinstance(module, Data2VecVisionLayer):
            if module.lambda_1 is not None:
                init.constant_(module.lambda_1, self.config.layer_scale_init_value)
                init.constant_(module.lambda_2, self.config.layer_scale_init_value)


@auto_docstring
# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitModel with BEIT->DATA2VEC_VISION,Beit->Data2VecVision,True->False
class Data2VecVisionModel(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig, add_pooling_layer: bool = False) -> None:
        r"""
        add_pooling_layer (bool, *optional*, defaults to `False`):
            Whether to add a pooling layer
        """
        super().__init__(config)
        self.config = config

        self.embeddings = Data2VecVisionEmbeddings(config)
        self.encoder = Data2VecVisionEncoder(config, window_size=self.embeddings.patch_embeddings.patch_shape)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = Data2VecVisionPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: torch.BoolTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | Data2VecVisionModelOutputWithPooling:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        embedding_output, _ = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        resolution = pixel_values.shape[2:]

        encoder_outputs = self.encoder(
            embedding_output,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            resolution=resolution,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return Data2VecVisionModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Copied from transformers.models.beit.modeling_beit.BeitPooler with Beit->Data2VecVision
class Data2VecVisionPooler(nn.Module):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Mean pool patch tokens with layernorm, or take the [CLS] token
        return self.layernorm(hidden_states[:, 1:, :].mean(1)) if self.layernorm is not None else hidden_states[:, 0]


@auto_docstring(
    custom_intro="""
    Data2VecVision Model transformer with an image classification head on top (a linear layer on top of the average of
    the final hidden states of the patch tokens) e.g. for ImageNet.
    """
)
# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitForImageClassification with BEIT->DATA2VEC_VISION,Beit->Data2VecVision,beit->data2vec_vision
class Data2VecVisionForImageClassification(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | ImageClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        outputs = self.data2vec_vision(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.beit.modeling_beit.BeitConvLayer with Beit->Data2VecVision
class Data2VecVisionConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        bias: bool = False,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.convolution(hidden_states)
        hidden_states = self.normalization(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


# Copied from transformers.models.beit.modeling_beit.BeitPyramidPoolingBlock with Beit->Data2VecVision
class Data2VecVisionPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(pool_scale)
        self.conv = Data2VecVisionConvLayer(in_channels, channels, kernel_size=1)

    def forward(self, input: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        hidden_state = self.pooling(input)
        hidden_state = self.conv(hidden_state)
        hidden_state = nn.functional.interpolate(hidden_state, size=size, mode="bilinear", align_corners=False)
        return hidden_state


# Copied from transformers.models.beit.modeling_beit.BeitPyramidPoolingModule with Beit->Data2VecVision
class Data2VecVisionPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales: tuple[int, ...], in_channels: int, channels: int) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = nn.ModuleList(
            [
                Data2VecVisionPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
                for pool_scale in pool_scales
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        original_size = hidden_states.size()[2:]
        return [block(hidden_states, size=original_size) for block in self.blocks]


# Copied from transformers.models.beit.modeling_beit.BeitUperHead with Beit->Data2VecVision
class Data2VecVisionUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://huggingface.co/papers/1807.10221).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = Data2VecVisionPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
        )
        self.psp_bottleneck = Data2VecVisionConvLayer(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            self.lateral_convs.append(Data2VecVisionConvLayer(in_channels, self.channels, kernel_size=1))
            self.fpn_convs.append(Data2VecVisionConvLayer(self.channels, self.channels, kernel_size=3, padding=1))

        self.fpn_bottleneck = Data2VecVisionConvLayer(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        hidden_state = hidden_states[-1]
        hidden_state = torch.cat([hidden_state, *self.psp_modules(hidden_state)], dim=1)
        return self.psp_bottleneck(hidden_state)

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        # build laterals
        laterals = []
        for lateral_conv, hidden_state in zip(self.lateral_convs, encoder_hidden_states):
            laterals.append(lateral_conv(hidden_state))

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=False
            )

        # build outputs
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            fpn_outs.append(self.fpn_convs[i](laterals[i]))
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


# Copied from transformers.models.beit.modeling_beit.BeitFCNHead with Beit->Data2VecVision
class Data2VecVisionFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of
    [FCNNet](https://huggingface.co/papers/1411.4038>).

    Args:
        config (Data2VecVisionConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(
        self,
        config: Data2VecVisionConfig,
        in_index: int = 2,
        kernel_size: int = 3,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        self.convs = nn.ModuleList()
        if self.num_convs > 0:
            self.convs.append(
                Data2VecVisionConvLayer(
                    self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
            for _ in range(self.num_convs - 1):
                self.convs.append(
                    Data2VecVisionConvLayer(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                    )
                )
        if self.concat_input:
            self.conv_cat = Data2VecVisionConvLayer(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states: list[torch.Tensor]) -> torch.Tensor:
        residual = encoder_hidden_states[self.in_index]
        hidden_states = residual
        for conv in self.convs:
            hidden_states = conv(hidden_states)
        if self.concat_input:
            hidden_states = self.conv_cat(torch.cat([residual, hidden_states], dim=1))
        hidden_states = self.classifier(hidden_states)
        return hidden_states


@auto_docstring
# Todo - Refactor as part of vision refactor. Copied from transformers.models.beit.modeling_beit.BeitForSemanticSegmentation with BEIT->DATA2VEC_VISION,Beit->Data2VecVision,microsoft/beit-base-finetuned-ade-640-640->facebook/data2vec-vision-base,beit->data2vec_vision
class Data2VecVisionForSemanticSegmentation(Data2VecVisionPreTrainedModel):
    def __init__(self, config: Data2VecVisionConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.data2vec_vision = Data2VecVisionModel(config, add_pooling_layer=False)

        # FPNs
        if len(self.config.out_indices) != 4:
            raise ValueError(
                "Data2VecVisionForSemanticSegmentation requires config.out_indices to be a list of 4 integers, "
                "specifying which features to use from the backbone. One can use [3, 5, 7, 11] in case of "
                "a base-sized architecture."
            )
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        self.decode_head = Data2VecVisionUperHead(config)
        self.auxiliary_head = Data2VecVisionFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # compute weighted loss
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        loss = main_loss
        if auxiliary_logits is not None:
            auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
            loss += self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, Data2VecVisionForSemanticSegmentation
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> image_processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        >>> model = Data2VecVisionForSemanticSegmentation.from_pretrained("facebook/data2vec-vision-base")

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> # logits are of shape (batch_size, num_labels, height, width)
        >>> logits = outputs.logits
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        outputs = self.data2vec_vision(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        logits = self.decode_head(features)

        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )


__all__ = [
    "Data2VecVisionForImageClassification",
    "Data2VecVisionForSemanticSegmentation",
    "Data2VecVisionModel",
    "Data2VecVisionPreTrainedModel",
]
