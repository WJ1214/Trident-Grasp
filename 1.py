from data.CornellDataset import CornellDataset
import torch
import torch.utils.data as Data
import cv2
import numpy as np
import data.util as util
import math
from Module.utils.bbox_tools import generate_base_anchor, enumerate_shifted_anchor
import torch.nn as nn
from Module.nms.NMS import nms
from collections import namedtuple
from torchvision.models import resnet50
from Module.TestModule.testnet import resnet18
from dcn.modules.deform_conv import DeformConvPack


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


# net = resnet18()
# print(net)
conv = DeformConvPack(3, 5, 3, 1, 1)
print(conv.weight)
print(conv.bias)




