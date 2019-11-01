from CornellDataset import CornellDataset
import torch.utils.data as Data
import numpy as np
import torch, torchvision
import os.path
import PIL.Image as Image
import cv2

root = 'D:\\Cornell_data\\dataset'
dataset = CornellDataset(root)
dataloader = Data.DataLoader(dataset, batch_size=5)
for data in dataloader:
    img, label = data
    print(img.shape)
    print(label)




