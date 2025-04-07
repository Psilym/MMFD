# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0] #l
    y1 = dets[:, 1] #b
    x2 = dets[:, 2] #r
    y2 = dets[:, 3] #t
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        ovr = inter / union
        # include remove
        compare_areas = np.expand_dims(areas[order[1:]],axis=-1)
        compare_areas = np.repeat(compare_areas,2,axis=-1)
        compare_areas[:,0] = areas[i]
        min_areas = compare_areas.min(axis=-1)
        inter_o_area = inter / min_areas
        inds = np.where(np.logical_and((ovr <= thresh),(inter_o_area < 0.7)))[0]
        # inds = np.where((ovr <= thresh))[0]
        order = order[inds + 1]

    return keep

def edge_remove(dets,scale,th = 10):
    l = dets[:, 0]  # l
    b = dets[:, 1]  # b
    r = dets[:, 2]  # r
    t = dets[:, 3]  # t
    scores = dets[:, 4]
    h,w = scale
    not_keep = (t >= (h - th)) | \
               (b<=th) | \
               (r>= (w - th)) | \
               (l<th)
    keep = np.logical_not(not_keep)

    return keep