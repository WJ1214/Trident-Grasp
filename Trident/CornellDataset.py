from torch.utils.data import Dataset
import os.path
import numpy as np
import torchvision
import PIL.Image as Image


class CornellDataset(Dataset):
    """
    输入数据集的存储路径，将image和相应标签存储至self.data中
    input:
    path:输入的字符串常量
    train:选择是否是训练集数据
    return:
    self.image:由PIL.Image以RGB格式打开的图像转换而成的tensor
    self.data:每张image对应的标签，
    """
    def __init__(self, path, train=True, transforms=torchvision.transforms.ToTensor()):
        self.image = []
        self.data = []                        # label of every image
        self.transforms = transforms
        if os.path.exists(path):
            self.image_path = "{}\\image".format(path)
            self.data_path = "{}\\pos_label".format(path)
        else:
            raise Exception("noSuchFilePath")

        max_num = 0
        for image, pos_label in zip(os.scandir(self.image_path), os.scandir(self.data_path)):
            image_label = []
            for lines in open(pos_label):
                image_label.append(list(map(float, lines.split())))
            image_label = np.array(image_label)
            # image_label = torch.from_numpy(image_label)
            num = image_label.shape[0]
            num = int(num/4)
            max_num = max(max_num, num)
            image_label = image_label.reshape((num, 8))
            self.data.append(image_label)
            img = "{root}\\image\\{image_name}".format(root=path, image_name=image.name)
            self.image.append(img)
        self.data = np.array(self.data)

        for i, elm in enumerate(self.data):
            elm_num = elm.shape[0]
            need_num = max_num - elm_num
            zero = np.zeros((need_num, 8))
            elm = np.vstack((elm, zero))
            self.data[i] = elm

        # chose if train data
        if train:
            number = int(len(self.image) * 0.7)
            self.image = self.image[:number]
            self.data = self.data[:number]
        else:
            number = int(len(self.image) * 0.7)
            self.image = self.image[number:]
            self.data = self.data[number:]

    def __getitem__(self, index):
        img = self.image[index]
        img = Image.open(img).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, self.data[index]

    def __len__(self):
        return len(self.data)


