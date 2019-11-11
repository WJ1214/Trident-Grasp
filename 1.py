from data.CornellDataset import CornellDataset
import torch.utils.data as Data
import torchvision.transforms as tvtsf
import cv2
import numpy as np
import data.util as util
from data.util import numpy_box2list

root = 'D:/Cornell_data/dataset'
dataset = CornellDataset(root)
dataloader = Data.DataLoader(dataset)
for data in dataloader:
    img, box, angle = data
    print('image:', img)
    print('box:', box)
    print('angle:', angle)


# for j in range(len(bbox)):
#     for i in range(4):
#         cv2.circle(img, tuple(bbox[j][i]), 3, (255, 0, 0), -1)
#         cv2.imshow('1', img)
#         cv2.waitKey(0)




