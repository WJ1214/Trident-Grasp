from data.CornellDataset import CornellDataset
import torch
import data.util as util
from torchvision.models import resnet50


# 数据集测试
root = 'D:/Cornell_data/dataset'
trans = util.PreProcess()
dataset = CornellDataset(root, True, trans)
data = dataset.__getitem__(0)
img, box, angle = data
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
net = resnet50(False)
device = torch.device('cuda:0')



