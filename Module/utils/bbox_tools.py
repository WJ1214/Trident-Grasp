import math
import numpy as np
import torch


def rotate_anchor_box(anchor_box, base_size, angle):
    # 输入一个基本的anchor_box(numpy类型)
    center_point = (base_size/2, base_size/2)
    x2, y2 = center_point
    res = []
    for i in range(anchor_box.shape[0]):
        if i % 2 == 0:
            point = anchor_box[i:i+2]
            x1 = point[0]
            y1 = point[1]
            x = (x1 - x2) * math.cos(math.pi / 180.0 * angle) - (y1 - y2) * math.sin(math.pi / 180.0 * angle) + x2
            y = (x1 - x2) * math.sin(math.pi / 180.0 * angle) + (y1 - y2) * math.cos(math.pi / 180.0 * angle) + y2
            res.append(x)
            res.append(y)
    return np.array(res)


def generate_base_anchor(base_size=32, angle=30):
    vertical_anchor = np.array([0, 0, base_size-1, 0, base_size-1, base_size-1, 0, base_size-1])
    base_anchor = rotate_anchor_box(vertical_anchor, base_size, angle)
    base_anchor = base_anchor.reshape((1, 8))
    return base_anchor, angle


def enumerate_shifted_anchor(base_anchor, feat_stride, height, width):
    # 输入经过特征卷积后的特征图的宽和高
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel()), axis=1)
    anchor_box = base_anchor[0] + shift
    return anchor_box
