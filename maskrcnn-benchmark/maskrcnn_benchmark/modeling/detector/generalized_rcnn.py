# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads

from torchvision.utils import save_image
from tqdm import tqdm


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg)
        self.flag=False

    def forward(self, images, targets=None):
        # for target in targets:
        #     print(target.get_field('rotations'), '==========')
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        imagen_tensor=images.tensors
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if not self.flag:
            self.save_feature_map(features,imagen_tensor)    #=============================================== SAVING FEATURE MAPS
        # print(images.tensors.size(),'=========================================')
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def save_feature_map(self,feature_maps,image_tensor):
        for ll,layer in  enumerate(tqdm(feature_maps)):
            if True:
                for mm,map in enumerate(layer):
                    save_image(image_tensor[mm],
                               './fm/layer' + str(ll) + 'image' + str(mm) +  '_.png')
                    if mm==0:
                        for cc,channel in enumerate(map):
                            if cc==81 or cc==89:
                                save_image(channel,'./fm/layer'+str(ll)+'image'+str(mm)+'channel'+str(cc)+'.png')
                                self.flag=True
                                # if cc>=5:
                                #     break

