from torch.utils.data import Dataset
import os.path
import numpy as np
import torchvision
import PIL.Image as Image
import cv2
import data.util as util


class CornellDataset(Dataset):
    """
    输入数据集的存储路径，将image和相应标签存储至self.data中
    input:
    path:输入的字符串常量
    train:选择是否是训练集数据
    return:
    self.image:由opencv以RGB格式打开的图像转换而成的tensor
    self.data:每张image对应的标签,list类型
    """
    def __init__(self, path, train=True, transforms=torchvision.transforms.ToTensor()):
        self.image = []
        self.data = []                        # bbox of every image
        self.angles = []
        self.transforms = transforms
        if os.path.exists(path):
            self.image_path = "{}/image".format(path)
            self.data_path = "{}/pos_label".format(path)
        else:
            raise Exception("noSuchFilePath")

        max_num = 0
        for image, pos_label in zip(os.scandir(self.image_path), os.scandir(self.data_path)):
            image_box = []
            for lines in open(pos_label):
                image_box.append(list(map(float, lines.split())))
            image_box = np.array(image_box)
            num = image_box.shape[0]
            num = int(num/4)
            max_num = max(max_num, num)
            image_box = image_box.reshape((num, 8))

            self.data.append(image_box)
            img = "{root}/image/{image_name}".format(root=path, image_name=image.name)
            self.image.append(img)

        box = self.data.copy()
        for bbox in box:
            angles = []
            bbox = util.numpy_box2list(bbox)
            for i, elm in enumerate(bbox):
                angle = util.calculate_angle(bbox[i])
                angles.append(angle)
            self.angles.append(angles)

        # chose if train data
        if train:
            number = int(len(self.image) * 0.7)
            self.image = self.image[:number]
            self.data = self.data[:number]
            self.angles = self.angles[:number]
        else:
            number = int(len(self.image) * 0.7)
            self.image = self.image[number:]
            self.data = self.data[number:]
            self.angles = self.angles[:number]

    def __getitem__(self, index):
        """

        :param index:
        :return: img:RGB格式的opencv打开文件
                self.data[index]:img相应的numpy格式抓取框
        """
        img = self.image[index]
        # img = Image.open(img).convert('RGB')
        img = cv2.imread(img)
        B, G, R = cv2.split(img)
        img = cv2.merge([R, G, B])
        # 将opencv的BGR格式转换为RGB格式
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.data[index], self.angles[index]

    def __len__(self):
        return len(self.data)


