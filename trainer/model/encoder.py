from typing import Tuple, List, Optional, Dict
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer.subsampling import LinearNoSubsampling
from layer.subsampling import Conv2dSubsampling4
from layer.subsampling import Conv2dSubsampling6
from layer.subsampling import Conv2dSubsampling8
from layer.positional_encoding import PositionalEncoding
from layer.positional_encoding import RelPositionalEncoding
from layer.positional_encoding import NoPositionalEncoding
from utils.mask import make_pad_mask
from utils.mask import add_optional_chunk_mask
from utils.common import log_add
from loss.loss_compute import CTCLoss


class BaseCTCEncoder(nn.Module):
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
        conv_subsample_in_ch: int = 1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blank_idx = blank_idx
        self.attention_dim = attention_dim
        # subsampling
        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
            subsampling_args = (input_dim, attention_dim, dropout_rate)
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
            assert input_dim % conv_subsample_in_ch == 0
            subsampling_args = (input_dim // conv_subsample_in_ch, attention_dim, conv_subsample_in_ch)
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.subsampling = subsampling_class(*subsampling_args)
        # positional embeding
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        self.pos_enc = pos_enc_class(attention_dim, positional_dropout_rate)
        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(attention_dim, eps=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        # encoder blocks should be defined by specific subclass
        # output layer
        self.out_linear = nn.Linear(attention_dim, self.output_dim)
        # criterion
        self.ctc_criterion = CTCLoss(self.blank_idx, mean_in_batch=True)

    @property
    def encoder_embed_dim(self):
        return self.attention_dim

    @property
    def metric_tags(self):
        tags = ['ctc_loss']
        return tags

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        given_chunk_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
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
            given_chunk_mask: use consistent chunk_mask with another process
        """
        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, T)
        xs, masks = self.subsampling(xs, masks)
        xs, pos_emb = self.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        if given_chunk_mask is not None:
            chunk_masks = given_chunk_mask
        else:
            chunk_masks = add_optional_chunk_mask(
                    xs, masks, self.use_dynamic_chunk,
                    self.use_dynamic_left_chunk, decoding_chunk_size,
                    self.static_chunk_size, num_decoding_left_chunks)
        for layer in self.blocks:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        out_nosm = self.out_linear(xs)
        out_lens = masks.sum(dim=-1).view(-1)
        res = {
            "out_nosm": out_nosm,
            "out_lens": out_lens,
            "hidden": xs
        }
        return res

    def cal_loss(self, res, target, target_lens):
        out_nosm = res['out_nosm']
        out_lens = res['out_lens']
        loss, metric, count = self.ctc_criterion(
            out_nosm, out_lens, target, target_lens)
        return loss, metric, count

    def ctc_greedy_search(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1
    ) -> List[List[int]]:
        # decoding does not support dynamic chunks
        assert decoding_chunk_size != 0
        res = self.forward(xs, xs_lens, decoding_chunk_size,
                           num_decoding_left_chunks)
        out_nosm, out_lens = res['out_nosm'], res['out_lens']
        bsz = out_nosm.size(0)
        argmax = out_nosm.argmax(dim=-1).cpu().numpy()
        out_lens = out_lens.cpu().numpy()
        hyps = []
        for i in range(bsz):
            hyp = []
            pre = -1
            for ele in argmax[i][0:out_lens[i]]:
                if ele != pre and ele != self.blank_idx:
                    hyp.append(ele)
                pre = ele
            hyps.append(hyp)
        return hyps

    def ctc_prefix_beam_search(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1
    ) -> Tuple[List[Tuple[List[int], float]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            xs (torch.Tensor): (batch, max_len, feat_dim)
            xs_lens (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            num_decoding_left_chunks: number of left chunks
        Returns:
            List[Tuple[List[int], float]]: nbest results with ctc scores
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert xs.shape[0] == xs_lens.shape[0]
        assert decoding_chunk_size != 0
        batch_size = xs.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        res = self.forward(xs, xs_lens, decoding_chunk_size,
                           num_decoding_left_chunks)
        out_nosm = res['out_nosm']
        encoder_out = res['hidden']
        max_len = out_nosm.size(1)
        ctc_probs = F.log_softmax(out_nosm, dim=-1)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, max_len):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            _, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == self.blank_idx:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out
