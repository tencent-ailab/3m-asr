from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from layer.decoder import TransformerDecoder
from layer.decoder import BiTransformerDecoder
from loss.loss_compute import LabelSmoothingLoss
from utils.common import add_sos_eos, reverse_pad_list
from utils.mask import make_pad_mask


class JointCtcAedModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_conf: Optional[Dict[str, Any]] = None,
        decoder_type: str = "transformer",
        decoder_conf: Optional[Dict[str, Any]] = None,
        ignore_id: int = -1,
        ctc_weight: float = 0.3,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalize_loss: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        vocab_size = output_dim
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.reverse_weight = reverse_weight
        self.ignore_id = ignore_id
        assert 0.0 <= ctc_weight <= 1.0
        self.ctc_weight = ctc_weight
        self.lsm_weight = lsm_weight
        self.length_normalize_loss = length_normalize_loss

    def build_decoder(
        self,
        vocab_size: int,
        encoder_out_dim: int,
        decoder_type: str = "transformer",
        decoder_conf: Optional[Dict[str, Any]] = None,
    ):
        if decoder_conf is None:
            # use default config of construction function
            decoder_conf = {}
        if decoder_type == "transformer":
            self.decoder = TransformerDecoder(
                vocab_size, encoder_out_dim, **decoder_conf)
        else:
            assert 0.0 < self.reverse_weight < 1.0
            assert "r_num_blocks" in decoder_conf and \
                    decoder_conf["r_num_blocks"] > 0
            self.decoder = BiTransformerDecoder(
                    vocab_size, encoder_out_dim, **decoder_conf)

    def build_criterion(self):
        # ctc
        self.ctc_criterion = self.encoder.ctc_criterion
        # aed
        self.att_criterion = LabelSmoothingLoss(
            self.output_dim, self.ignore_id, self.lsm_weight,
            normalize_length=self.length_normalize_loss)

    def forward(
        self,
        feats: torch.Tensor,
        feat_lens: torch.Tensor,
        target: torch.Tensor,
        target_lens: torch.Tensor
    ) -> Dict[str, Any]:
        # ctc branch
        res = self.encoder(feats, feat_lens)
        encoder_out = res['hidden']
        out_lens = res['out_lens']
        max_step = encoder_out.size(1)
        encoder_mask = ~make_pad_mask(out_lens, max_step)
        encoder_mask = encoder_mask.unsqueeze(1)
        # aed branch
        ys_in_pad, ys_out_pad = add_sos_eos(target, self.sos, self.eos, self.ignore_id)
        ys_in_lens = target_lens + 1
        # reverse the seq, used for right-to-left decoder
        r_ys_pad = reverse_pad_list(target, target_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos, self.ignore_id)
        # forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, ys_in_pad,
            ys_in_lens, r_ys_in_pad, self.reverse_weight)
        res['decoder_out'] = decoder_out
        res['r_decoder_out'] = r_decoder_out
        res['ys_out_pad'] = ys_out_pad
        res['r_ys_out_pad'] = r_ys_out_pad
        return res

    @property
    def metric_tags(self):
        tags = []
        if self.ctc_weight > 0.0:
            tags += ['ctc_loss']
        if self.ctc_weight < 1.0:
            tags += ['aed_loss']
        return tags

    def cal_loss(self, res, target, target_lens):
        out_nosm = res['out_nosm']
        out_lens = res['out_lens']
        decoder_out, ys_out_pad = res['decoder_out'], res['ys_out_pad']
        r_decoder_out, r_ys_out_pad = res['r_decoder_out'], res['r_ys_out_pad']
        loss, metric, count = 0.0, (), ()
        if self.ctc_weight > 0.0:
            loss_ctc, metric_ctc, count_ctc = self.ctc_criterion(
                out_nosm, out_lens, target, target_lens)
            loss += self.ctc_weight * loss_ctc
            metric += metric_ctc
            count += count_ctc
        if self.ctc_weight < 1.0:
            loss_att, metric_att, count_att = self.att_criterion(
                decoder_out, ys_out_pad)
            # ignore the metric of reverse decoder
            if hasattr(self.decoder, "right_decoder"):
                r_loss_att, _, _ = self.att_criterion(
                    r_decoder_out, r_ys_out_pad)
                loss_att = loss_att * (1 - self.reverse_weight) + \
                            self.reverse_weight * r_loss_att
            loss += (1 - self.ctc_weight) * loss_att
            metric += metric_att
            count += count_att
        return loss, metric, count

    def forward_encoder(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def ctc_greedy_search(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1
    ) -> List[List[int]]:
        return self.encoder.ctc_greedy_search(
                xs, xs_lens, decoding_chunk_size, num_decoding_left_chunks)

    def ctc_prefix_beam_search(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1
    ) -> Tuple[List[Tuple[List[int], float]], torch.Tensor]:
        return self.encoder.ctc_prefix_beam_search(
                xs, xs_lens, beam_size, decoding_chunk_size,
                num_decoding_left_chunks)

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out = self.ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks)
        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = F.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = F.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0]
