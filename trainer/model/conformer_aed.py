from typing import Optional, Dict, Any
from model.conformer import Net as ConformerEncoder
from model.ctc_aed import JointCtcAedModel


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
        self.encoder = ConformerEncoder(input_dim, output_dim, **encoder_conf)
        encoder_out_dim = self.encoder.encoder_embed_dim
        self.build_decoder(self.output_dim, encoder_out_dim,
                decoder_type, decoder_conf)
        self.build_criterion()
