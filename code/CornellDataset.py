from torch.utils.data import Dataset
import os.path
import numpy as np


class CornellDataset(Dataset):
    """
    输入数据集的存储路径，将image和相应标签存储至self.data中
    path:输入的字符串常量
    image:数据集的图像(没有经过任何转换)
    image_label:每张image对应的标签，
    """
    def __init__(self, path):
        self.data = []
        if os.path.exists(path):
            self.image_path = "{}\\image".format(path)
            self.data_path = "{}\\pos_label".format(path)
        else:
            raise Exception("noSuchFilePath")

        for image, pos_label in zip(os.scandir(self.image_path), os.scandir(self.data_path)):
            image_label = []
            for lines in open(pos_label):
                image_label.append(list(map(float, lines.split())))
            image_label = np.array(image_label)
            self.data.append([image, image_label])

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


