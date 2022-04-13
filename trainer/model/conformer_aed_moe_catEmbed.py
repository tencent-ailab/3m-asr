from typing import Optional, Dict, Any
from model.ctc_aed import JointCtcAedModel
from model.conformer_moe_catEmbed import Net as ConformerMoeEncoder


class Net(JointCtcAedModel):
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
        super().__init__(
            input_dim, output_dim, encoder_conf, decoder_type,
            decoder_conf, ignore_id, ctc_weight, reverse_weight,
            lsm_weight, length_normalize_loss)
        # use default config for construction function
        if encoder_conf is None:
            encoder_conf = {}
        if decoder_conf is None:
            decoder_conf = {}
        self.encoder = ConformerMoeEncoder(input_dim, output_dim, **encoder_conf)
        encoder_out_dim = self.encoder.encoder_embed_dim
        self.moe_conf = self.encoder.moe_conf
        self.build_decoder(self.output_dim, encoder_out_dim,
                decoder_type, decoder_conf)
        self.build_criterion()
        self.aux_criterion = self.encoder.aux_criterion

    def init_embed_model(self, load_path):
        return self.encoder.init_embed_model(load_path)

    def init_experts_from_base(self, load_path):
        return ConformerMoeEncoder.init_experts_from_base(self, load_path)

    @property
    def metric_tags(self):
        tags = super().metric_tags
        encoder_tags = self.encoder.metric_tags
        encoder_tags.pop(0)  # pop the redundant 'ctc_loss' tag
        tags += encoder_tags
        return tags

    def cal_loss(self, res, target, target_lens):
        loss, metric, count = super().cal_loss(res, target, target_lens)
        # other loss
        out_lens = res['out_lens']
        embed_out_nosm = res['embed_out_nosm']
        aux_loss = res['aux_loss']
        if self.encoder.embed_scale > 0.0:
            loss_embed, metric_embed, count_embed = self.ctc_criterion(
                embed_out_nosm, out_lens, target, target_lens)
            loss += self.encoder.embed_scale * loss_embed
            metric += metric_embed
            count += count_embed
        loss_aux, metric_aux, count_aux = self.aux_criterion(aux_loss)
        loss += loss_aux
        metric += metric_aux
        count += count_aux
        return loss, metric, count

    def state_dict_comm(self):
        return ConformerMoeEncoder.state_dict_comm(self)

    def load_state_dict_comm(self, state_dict):
        return ConformerMoeEncoder.load_state_dict_comm(self, state_dict)
