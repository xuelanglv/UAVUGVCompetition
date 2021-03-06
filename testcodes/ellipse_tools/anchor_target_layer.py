# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
import numpy as np
import numpy.random as npr


import caffe
from fast_rcnn.config import cfg
from ellipse_tools import generate_anchors
from ellipse_tools.rbbox_transform import rbbox_transform
from ellipse_tools.rbbox_angle_distance import angle_distance
from ellipse_tools.rbbox_overlaps import rbbx_overlaps
from ellipse_tools.inside_judge import ind_inside, condinate_rotate


from compiler.ast import flatten


DEBUG = False


class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        anchor_scales = layer_params.get('scales', (8, 16, 32))
        self._anchors = generate_anchors(scales=np.array(anchor_scales))
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = layer_params['feat_stride']

        # [x_ctr, y_ctr, height, width, theta] anti-clock-wise angle
        self.bbox_para_num = 5

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]

        A = self._num_anchors

        # labels
        top[0].reshape(1, 1, A * height, width)
        # bbox_targets
        top[1].reshape(1, A * self.bbox_para_num, height, width)
        # bbox_inside_weights
        top[2].reshape(1, A * self.bbox_para_num, height, width)
        # bbox_outside_weights
        top[3].reshape(1, A * self.bbox_para_num, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        assert bottom[0].data.shape[0] == 1, \
            'Only single item batches are supported'

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (x_ctr, y_ctr, height, width, theta, label)
        gt_boxes = bottom[1].data
        # im_info
        im_info = bottom[2].data[0, :]


        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), np.zeros((3, width * height)))).transpose()
        # add A anchors (1, A, 5) to
        # cell K shifts (K, 1, 5) to get
        # shift anchors (K, A, 5)
        # reshape to (K*A, 5) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, self.bbox_para_num)) + 
                       shifts.reshape((1, K, self.bbox_para_num)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, self.bbox_para_num))
        total_anchors = int(K * A)


        # 2. Keep the anchors inside the image
        # only keep anchors inside the image
        pt1, pt2, pt3, pt4 = condinate_rotate(all_anchors)  # coodinate project
        inds_inside = np.array(ind_inside(pt1, pt2, pt3, pt4, im_info[0], im_info[1]))  # inside index
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # 3, Make Label:1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1) # the default label of all anchors are -1

        # calculate the overlaps between anchors and gt_boxes
        overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32), np.ascontiguousarray(gt_boxes[:, 0:5], dtype=np.float32), cfg.GPU_ID)

        an_gt_diffs = angle_distance(anchors, gt_boxes)

        argmax_overlaps = overlaps.argmax(axis=1)  # max overlaps of anchor compared with gts
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        max_overlaps_angle_diff = an_gt_diffs[np.arange(len(inds_inside)), argmax_overlaps]  # D

        gt_argmax_overlaps = overlaps.argmax(axis=0)  # max overlaps of gt compared with anchors
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where((overlaps == gt_max_overlaps) & (an_gt_diffs <= cfg.TRAIN.R_POSITIVE_ANGLE_FILTER))[0]


        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0


        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # D
        # fg label: above threshold IOU the angle diff abs must be less than 15

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[(max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP) | ((max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP) & (max_overlaps_angle_diff > cfg.TRAIN.R_NEGATIVE_ANGLE_FILTER))] = 0
	
        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1


        bbox_targets = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_RBBOX_INSIDE_WEIGHTS)  # D

        bbox_outside_weights = np.zeros((len(inds_inside), self.bbox_para_num), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, self.bbox_para_num)) * 1.0 / num_examples
            negative_weights = np.ones((1, self.bbox_para_num)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) & 
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT / 
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) / 
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights


        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights.shape[2] == height
        assert bbox_inside_weights.shape[3] == width
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * self.bbox_para_num)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights
        
        '''
        print 'rpn_labels_shape', top[0].data.shape
        print 'rpn_labels' , top[0].data
        print 'flatten', top[0].data.flatten()
        print 'rpn_valid_data', np.where(top[0].data.flatten() > 0)
        print 'rpn_bbox_targets', top[1].data
        '''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 5  # 
    assert gt_rois.shape[1] == 6

    return rbbox_transform(ex_rois, gt_rois[:, :5]).astype(np.float32, copy=False)


if __name__ == "__main__":
    
    anchors = np.array([[100,100,100,100,0],[100,100,50,50,0]])
    gt_boxes = np.array([[100,100,200,200,0]])
    
    overlaps = rbbx_overlaps(np.ascontiguousarray(anchors, dtype=np.float32), np.ascontiguousarray(gt_boxes[:, 0:5], dtype=np.float32), cfg.GPU_ID)
    overlaps = overlaps[:,0]
    S_img = gt_boxes[0,2] * gt_boxes[0,3]
    W = overlaps/(1+overlaps)
    S_elp = anchors[:,2] * anchors[:,3]
    
    print 'overlaps:', overlaps
    print 'over area:', W * (S_elp + S_img)
    print W
    print S_img
    print S_elp
    print W * (S_elp + S_img)/S_elp
