# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains specific functions for computing losses on the RPN
file
"""

import torch
from torch.nn import functional as F

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_anchors(self, anchor, target):
        # print(len(anchor),len(target),'==============================match=====')
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(['rotations'])
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        orien_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            # print(len(anchors_per_image), len(targets_per_image), '===============prepare====================')
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)
            # discard anchors that go out of the boundaries of the image
            labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )
            # compute orientation targets=======================================
            orien_targets_per_image = matched_targets.get_field("rotations")
            # orien_targets_per_image = orien_targets_per_image.to(dtype=torch.int64)

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            orien_targets.append(orien_targets_per_image)

        return labels, regression_targets, orien_targets

    def __call__(self, anchors, objectness, box_regression, box_orien, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        # print(targets,'===================================')
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets, orien_targets = self.prepare_targets(anchors, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness_flattened = []
        box_regression_flattened = []
        box_orien_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the objectness and the box_regression
        for objectness_per_level, box_regression_per_level, box_orien_per_level in zip(
                objectness, box_regression, box_orien
        ):
            N, A, H, W = objectness_per_level.shape
            # print(box_orien_per_level.shape)
            objectness_per_level = objectness_per_level.permute(0, 2, 3, 1).reshape(
                N, -1
            )

            box_regression_per_level = box_regression_per_level.view(N, -1, 4, H, W)
            box_regression_per_level = box_regression_per_level.permute(0, 3, 4, 1, 2)
            box_regression_per_level = box_regression_per_level.reshape(N, -1, 4)
            box_orien_per_level = box_orien_per_level.view(N, -1, 2, H, W)
            box_orien_per_level = box_orien_per_level.permute(0, 3, 4, 1, 2)
            box_orien_per_level = box_orien_per_level.reshape(N, -1, 2)
            # print(box_regression_per_level.shape)
            # print(box_orien_per_level.shape,'========================')
            objectness_flattened.append(objectness_per_level)
            box_regression_flattened.append(box_regression_per_level)
            box_orien_flattened.append(box_orien_per_level)
        # concatenate on the first dimension (representing the feature levels), to
        # take into account the way the labels were generated (with all feature maps
        # being concatenated as well)
        objectness = cat(objectness_flattened, dim=1).reshape(-1)
        box_regression = cat(box_regression_flattened, dim=1).reshape(-1, 4)
        orien_regression = cat(box_orien_flattened, dim=1).reshape(-1, 2)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        orien_targets = torch.cat(orien_targets, dim=0)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta=1.0 / 9,
            size_average=False,
        ) / (sampled_pos_inds.numel() + 0.1)

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds], labels[sampled_inds]
        )
        # print(orien_targets[sampled_pos_inds],'=========orien===========')
        # print(orien_regression[sampled_pos_inds].size(), '=========regression===========\n')

        orien_loss = torch.sqrt(F.mse_loss(
            orien_regression[sampled_pos_inds],
            orien_targets[sampled_pos_inds].type(torch.cuda.FloatTensor),
            # size_average=False,
            # beta=1,
        )) / (sampled_pos_inds.numel() + 0.1)

        # print(orien_loss)

        return objectness_loss, box_loss, orien_loss


def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(matcher, fg_bg_sampler, box_coder)
    return loss_evaluator
