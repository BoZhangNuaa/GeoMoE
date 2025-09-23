import copy
import torch
from mmdet.models.detectors import TwoStageDetector
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class FasterRCNN_AUX(TwoStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
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
            losses.update({'aux_loss': self.aux_loss})
            self.aux_loss = None

        return losses
