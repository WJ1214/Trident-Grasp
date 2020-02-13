from data.CornellDataset import CornellDataset
import torch
import data.util as util
from torchvision.models import resnet50
import os
import torch.utils.data as Data


# 数据集测试
root = '/Users/wangjie/Downloads/cornell data'
# trans = util.PreProcess()
dataset = CornellDataset(root)
length = dataset.__len__()
for i in range(length):
	print(i)
	print(dataset.data)

# data = dataset.data[0]
# img = dataset.image[0]
# print(img)
# print(data)
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


# test_dic = {'model': 1, 'path': 'home/Downloads', 'name': 'this model'}
# path = 'C:/Users/王杰/PycharmProjects/Trident-Grasp/save_test/savefile'
# torch.save(test_dic, path)








