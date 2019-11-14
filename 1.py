from data.CornellDataset import CornellDataset
import torch.utils.data as Data
import torchvision.transforms as transform
import cv2
import numpy as np
import data.util as util

root = 'D:/Cornell_data/dataset'
trans = util.RandomCenterCrop()
dataset = CornellDataset(root, True, trans)
dataset1 = CornellDataset(root, True)
# data = dataset.__getitem__(0)
# img, box, angle = data
# data = dataset.__getitem__(0)
# img, box, angle = data
# cv2.imshow('1', img)
# cv2.waitKey(0)

dataloader = Data.DataLoader(dataset, 1)
for i, data in enumerate(dataloader):
    img, box, angle = data
    util.show_batch_image(img, box)









