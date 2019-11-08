from data.CornellDataset import CornellDataset
import torch.utils.data as Data
import torchvision.transforms as tvtsf

root = 'D:/Cornell_data/dataset'
dataset = CornellDataset(root)
data = dataset.__getitem__(0)
img, label = data




