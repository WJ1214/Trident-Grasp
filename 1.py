from data.CornellDataset import CornellDataset
import torch.utils.data as Data
import torchvision.transforms as transform
import cv2
import numpy as np
import data.util as util
import matplotlib.pyplot as plt
from skimage import io, data, transform
import math
from Module.utils.bbox_tools import generate_base_anchor, enumerate_shifted_anchor
import torch


# 数据集测试
root = 'D:/Cornell_data/dataset'
trans = util.PreProcess()
dataset = CornellDataset(root, True, trans)
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
anchor_box, _ = generate_base_anchor(32, 150)
# x = torch.arange(0, 10 * 32, 32)
# y = torch.arange(0, 10 * 32, 32)
# x, y = np.meshgrid(x, y)
# shift = np.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel(), x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
# anchor_box = anchor_box[0] + shift
anchor_box = enumerate_shifted_anchor(anchor_box, 32, 10, 10)
anchor_box = np.array([anchor_box])
dataloader = Data.DataLoader(dataset, 1)
for i, data in enumerate(dataloader):
    img, box, angle = data
    util.show_batch_image(img, anchor_box)













