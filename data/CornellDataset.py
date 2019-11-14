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
        self.max_num = 0
        if os.path.exists(path):
            self.image_path = "{}/image".format(path)
            self.data_path = "{}/pos_label".format(path)
        else:
            raise Exception("noSuchFilePath")

        for image, pos_label in zip(os.scandir(self.image_path), os.scandir(self.data_path)):
            image_box = []
            for lines in open(pos_label):
                image_box.append(list(map(float, lines.split())))
            image_box = np.array(image_box)
            num = image_box.shape[0]
            num = int(num/4)
            self.max_num = max(self.max_num, num)
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
            angles = np.array(angles)
            self.angles.append(angles)

        # for i, (bbox, angle) in enumerate(zip(self.data, self.angles)):
        #     bbox = util.box_add0(bbox, max_num)
        #     angle = util.angle_add0(angle, max_num)
        #     self.data[i] = bbox
        #     self.angles[i] = angle

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
        if not isinstance(self.transforms, torchvision.transforms.ToTensor):
            img, box = self.transforms(img, self.data[index])
            trans = torchvision.transforms.ToTensor()
            img = trans(img)
            box = util.box_add0(box, self.max_num)
        else:
            img = self.transforms(img)
            box = self.data[index]
            box = util.box_add0(box, self.max_num)
        angle = util.angle_add0(self.angles[index], self.max_num)
        return img, box, angle

    def __len__(self):
        return len(self.data)


