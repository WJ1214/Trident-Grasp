from data.CornellDataset import CornellDataset
import torch
import torch.utils.data as Data
import torchvision.transforms as transform
import cv2
import numpy as np
import data.util as util
import math
from Module.utils.bbox_tools import generate_base_anchor, enumerate_shifted_anchor
import torch.nn as nn
from Module.nms.NMS import intersection, nms, cpu_nms, non_max_suppression


# 数据集测试
root = 'D:/Cornell_data/dataset'
trans = util.PreProcess()
# dataset = CornellDataset(root, True, trans)
# data = dataset.__getitem__(0)
# img, box, angle = data
# anchor_box, _ = generate_base_anchor()
# util.show_bbox_image(img, anchor_box)


# 预处理测试
# dataloader = Data.DataLoader(dataset, 1)
# for i, data in enumerate(dataloader):
#     img, box, angle = data
#     util.show_batch_image(img, box)


# base_anchor测试
# anchor_box, _ = generate_base_anchor(32, 0)
# anchor_box = enumerate_shifted_anchor(anchor_box, 32, 10, 10)
# anchor_box = np.array([anchor_box])
# dataloader = Data.DataLoader(dataset, 1)
# for i, data in enumerate(dataloader):
#     img, box, angle = data
#     util.show_batch_image(img, box)

test_data = [[204, 102, 358, 102, 358, 250, 204, 250, 0.5],
             [257, 118, 380, 118, 380, 250, 257, 250, 0.7],
             [280, 135, 400, 135, 400, 250, 280, 250, 0.6],
             [255, 118, 360, 118, 360, 235, 255, 235, 0.7]]
dets = np.array([
                [204, 102, 358, 250, 0.5],
                [257, 118, 380, 250, 0.7],
                [280, 135, 400, 250, 0.6],
                [255, 118, 360, 235, 0.7]])

data2 = [[10, 10, 15, 10, 15, 20, 10, 20, 0.8],
         [12, 13, 17, 13, 17, 23, 12, 23, 0.7],
         [50, 20, 60, 20, 60, 70, 50, 70, 0.6]]
test_data2 = [[10, 10, 15, 20, 0.8],
              [12, 13, 17, 23, 0.7],
              [50, 20, 60, 70, 0.6]]
data2 = np.array(data2)
test_data2 = np.array(test_data2)

test_data = np.array(test_data)
rotate_keep = nms(data2, 0.8)
keep = cpu_nms(test_data2, 0.8)
print(rotate_keep)
print(keep)


















