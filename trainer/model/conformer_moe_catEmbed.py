from typing import Tuple, List, Optional, Dict, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from model.encoder import BaseCTCEncoder
from model.conformer import Net as ConformerEmbed
from layer.encoder_layer import FmoeConformerLayer
from layer.positionwise_feed_forward import PositionwiseFeedForward
from layer.positionwise_feed_forward import FmoeCatEmbedFeedForward
from layer.attention import MultiHeadedAttention
from layer.attention import RelPositionMultiHeadedAttention
from layer.convolution import ConvolutionModule
from utils.common import get_activation
from utils.mask import make_pad_mask
from utils.mask import add_optional_chunk_mask
from loss.loss_compute import MoELayerScaleAuxLoss


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
        conv_subsample_in_ch: int = 1,
        embed_conf: Optional[Dict[str, Any]] = None,
        moe_conf: Optional[Dict[str, Any]] = None,
        embed_scale: float = 0.0,
        aux_scale: Optional[List[float]] = None
    ):
        super().__init__(
            input_dim, output_dim, blank_idx, attention_heads, attention_dim,
            linear_units, num_blocks, dropout_rate, positional_dropout_rate,
            attention_dropout_rate, input_layer, pos_enc_layer_type,
            normalize_before, concat_after, static_chunk_size,
            use_dynamic_chunk, use_dynamic_left_chunk, conv_subsample_in_ch)
        activation = get_activation(activation_type)
        # embedding network
        if embed_conf is None:
            embed_conf = {}  # use default config of construction function
        self.embed = ConformerEmbed(input_dim, output_dim, **embed_conf)
        self.embed_scale = embed_scale
        # moe conf
        self.moe_conf = {
            'rank': 0,
            'world_size': 1,
            'comm': None,
            'num_experts': 4,
            'hidden_units': 1024,
            'dropout_rate': 0.0,
            'activation': activation,
            'capacity_factor': -1.0,
            'router_regularization': 'l1_plus_importance',
            'router_with_bias': False,
            'keep_expert_output': False,
            'rand_init_router': False
        }
        if moe_conf is not None:
            self.moe_conf.update(moe_conf)
        if self.moe_conf['router_regularization'] == 'l1_plus_importance':
            num_aux = 2
            if aux_scale is not None:
                assert len(aux_scale) == num_aux
            else:
                aux_scale = [0.1] * num_aux
            self.aux_tags = ["sparse_loss", "balance_loss"]
            aux_minimum = [num_blocks] * num_aux
        else:
            raise NotImplementedError("router regularization {} not supported".format(
                                      self.moe_conf['router_regularization']))
        # aux criterion
        self.aux_criterion = MoELayerScaleAuxLoss(num_aux, aux_scale, aux_minimum)
        # attention layer
        if pos_enc_layer_type == "no_pos":
            selfattn_layer = MultiHeadedAttention
        else:
            selfattn_layer = RelPositionMultiHeadedAttention
        san_layer_args = (
            attention_heads,
            attention_dim,
            attention_dropout_rate,
        )
        # feed-forward module in conformer
        moe_positionwise_layer = FmoeCatEmbedFeedForward
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim,
            self.moe_conf['hidden_units'],
            self.moe_conf['dropout_rate'],
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
        embed_dim = self.embed.encoder_embed_dim
        self.blocks = torch.nn.ModuleList([
            FmoeConformerLayer(
                attention_dim,
                selfattn_layer(*san_layer_args),
                moe_positionwise_layer(attention_dim, embed_dim, **self.moe_conf),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])

    def init_embed_model(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        self.embed.load_state_dict(param_dict)

    def init_experts_from_base(self, load_path):
        param_dict = torch.load(load_path, map_location='cpu')
        model_dict = self.state_dict()
        load_param_list = []
        for k, v in model_dict.items():
            if k in param_dict and param_dict[k].size() == v.size():
                model_dict[k] = param_dict[k]
                load_param_list.append(k)
            elif "experts" in k:
                ori_k = k.replace("experts.", "")
                if ori_k in param_dict and param_dict[ori_k].size() == v.size()[1:]:
                    model_dict[k] = param_dict[ori_k].unsqueeze(0).expand(v.size())
                    load_param_list.append(k)
        load_param_list.sort()
        self.load_state_dict(model_dict)
        return load_param_list

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1
    ) -> Dict[str, Any]:
        """
        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        """
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        input_bk = xs
        xs, masks = self.subsampling(xs, masks)
        xs, pos_emb = self.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        # use the same chunk mask for embedding network
        embedding_res = self.embed(
            input_bk, xs_lens, given_chunk_mask=chunk_masks)
        embedding = embedding_res['hidden']
        embedding = embedding.detach()
        embed_out = embedding_res['out_nosm']

        aux_loss_collection = []
        for layer in self.blocks:
            xs, aux_loss_res, chunk_masks, _ = layer(
                xs, embedding, chunk_masks, pos_emb, mask_pad)
            for aux_loss in aux_loss_res:
                aux_loss_collection.append(aux_loss)
        if self.normalize_before:
            xs = self.after_norm(xs)
        out_nosm = self.out_linear(xs)
        out_lens = masks.sum(dim=-1).view(-1)
        res = {
            "out_nosm": out_nosm,
            "out_lens": out_lens,
            "hidden": xs,
            "embed_out_nosm": embed_out,
            "aux_loss": aux_loss_collection,
        }
        return res

    @property
    def metric_tags(self):
        tags = ['ctc_loss']
        if self.embed_scale > 0.0:
            tags += ['embed_ctc_loss']
        tags += self.aux_tags
        return tags

    def cal_loss(self, res, target, targer_lens):
        out_nosm = res['out_nosm']
        out_lens = res['out_lens']
        embed_out_nosm = res['embed_out_nosm']
        aux_loss = res['aux_loss']
        # ctc
        loss, metric, count = self.ctc_criterion(
            out_nosm, out_lens, target, target_lens)
        # embed ctc
        if self.embed_scale > 0.0:
            loss_embed, metric_embed, count_embed = self.ctc_criterion(
                embed_out_nosm, out_lens, target, target_lens)
            loss += self.embed_scale * loss_embed
            metric += metric_embed
            count += count_embed
        # aux loss for moe routers
        loss_aux, metric_aux, count_aux = self.aux_criterion(aux_loss)
        loss += loss_aux
        metric += metric_aux
        count += count_aux
        return loss, metric, count

    def state_dict_comm(self):
        local_state_dict = self.state_dict()
        rank = self.moe_conf['rank']
        world_size = self.moe_conf['world_size']
        num_experts = self.moe_conf['num_experts']
        comm = self.moe_conf['comm']
        if world_size <= 1:
            return local_state_dict
        else:
            new_state_dict = OrderedDict()
            all_experts_num = world_size * num_experts
            for k, v in local_state_dict.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    new_size = list(v.size())
                    new_size[0] = all_experts_num
                    experts_weight = v.data.new_zeros(*new_size)
                    experts_weight[rank * num_experts: (rank + 1) * num_experts] = v
                    dist.all_reduce(experts_weight, group=comm, async_op=False)
                    new_state_dict[k] = experts_weight
            return new_state_dict

    def load_state_dict_comm(self, whole_model_state):
        rank = self.moe_conf['rank']
        world_size = self.moe_conf['world_size']
        num_experts = self.moe_conf['num_experts']
        if world_size <= 1:
            return self.load_state_dict(whole_model_state)
        else:
            new_state_dict = OrderedDict()
            for k, v in whole_model_state.items():
                if "experts" not in k:
                    new_state_dict[k] = v
                else:
                    assert v.size(0) == num_experts * world_size
                    new_state_dict[k] = v[rank * num_experts: (rank + 1) * num_experts]
            return self.load_state_dict(new_state_dict)
