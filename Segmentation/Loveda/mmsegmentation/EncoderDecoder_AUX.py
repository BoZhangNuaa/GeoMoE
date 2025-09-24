import copy
import torch
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig
from typing import List, Optional


@MODELS.register_module()
class EncoderDecoder_AUX(EncoderDecoder):
    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.aux_loss = None

    def extract_feat(self, batch_inputs):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x, self.aux_loss = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs,
             batch_data_samples) -> dict:

        losses = super().loss(batch_inputs, batch_data_samples)

        if self.aux_loss:
            losses.update({'moe_aux_loss': self.aux_loss})
            self.aux_loss = None

        return losses
