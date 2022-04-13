import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder import BaseCTCEncoder
from layer.positionwise_feed_forward import PositionwiseFeedForward
from layer.encoder_layer import ConformerEncoderLayer
from layer.attention import MultiHeadedAttention
from layer.attention import RelPositionMultiHeadedAttention
from layer.convolution import ConvolutionModule
from utils.common import get_activation


class Net(BaseCTCEncoder):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        blank_idx: int = 0,
        attention_heads: int = 4,
        attention_dim: int = 256,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        conv_subsample_in_ch: int = 1
    ):
        super().__init__(
            input_dim, output_dim, blank_idx, attention_heads, attention_dim,
            linear_units, num_blocks, dropout_rate, positional_dropout_rate,
            attention_dropout_rate, input_layer, pos_enc_layer_type,
            normalize_before, concat_after, static_chunk_size,
            use_dynamic_chunk, use_dynamic_left_chunk, conv_subsample_in_ch)
        # attention layer
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "no_pos":
            selfattn_layer = MultiHeadedAttention
        else:
            selfattn_layer = RelPositionMultiHeadedAttention
        san_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
        )
        # feed-forward module in attention
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module in attention
        convolution_layer = ConvolutionModule
        convolution_layer_args = (
            attention_dim,
            cnn_module_kernel,
            activation,
            cnn_module_norm,
            causal,
        )
        # encoder blocks
        self.blocks = nn.ModuleList([
            ConformerEncoderLayer(
                attention_dim,
                selfattn_layer(*san_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
