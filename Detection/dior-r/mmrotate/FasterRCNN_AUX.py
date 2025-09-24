import copy
import torch
from mmdet.models.detectors import TwoStageDetector
from mmrotate.registry import MODELS


@MODELS.register_module()
class FasterRCNN_AUX(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 data_preprocessor=None,
                 init_cfg=None) -> None:
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
