from CornellDataset import CornellDataset
import torch.utils.data as Data
import numpy as np
import torch

# dataset = CornellDataset("D:\\Cornell_data\\01")
# dataloader = Data.DataLoader(dataset=dataset)
#
#
# data = []
# image_path = "D:\\Cornell_data\\01\\image"
# data_path = "D:\\Cornell_data\\01\\pos_label"
# for image, pos_label in zip(os.scandir(image_path), os.scandir(data_path)):
#     image_label = []
#     for lines in open(pos_label):
#         image_label.append(list(map(float, lines.split())))
#     image_label = np.array(image_label)
#     data.append([image, image_label])
#
# print(data[:1])
a = np.array([[1, 2, 3, 4], [2, 3, 4, 1], [4, 5, 6, 7]])
b = np.array([[0, 2, 1], [1, 0, 2], [2, 0, 1]])
a = torch.from_numpy(a)
b = torch.from_numpy(b)
print(a[:, b.long()][torch.arange(3), torch.arange(3)])
