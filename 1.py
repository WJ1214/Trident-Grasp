from data.CornellDataset import CornellDataset
import torch.utils.data as Data
import torchvision.transforms as transform
import cv2
import numpy as np
import data.util as util
import matplotlib.pyplot as plt
from skimage import io, data, transform
import math

root = 'D:/Cornell_data/dataset'
trans = util.PreProcess()
dataset = CornellDataset(root, True, trans)
data = dataset.__getitem__(0)
img, box, angle = data
# cv2.imshow('1', img)
# cv2.waitKey(0)

dataloader = Data.DataLoader(dataset, 5)
for i, data in enumerate(dataloader):
    img, box, angle = data
    util.show_batch_image(img, box)


# img_path = dataset.image[0]
# img = io.imread(img_path)
# img = img[80:400, 160:480, :]
# point = np.array((110, 175))
#
# plt.imshow(img)
# plt.scatter(point[0], point[1], color='b', marker='.')
# plt.show()
#
# point1 = calculate_loc(point, img, 30)
# img = transform.rotate(img, 30, False)
# plt.imshow(img)
# plt.scatter(point1[0], point1[1], color='b', marker='.')
# plt.show()












