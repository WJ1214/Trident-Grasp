from TridentNet import TridentNet
from Blocks import TridentBlock
import torch.nn as nn
import torchvision.models as models


def tridentnet50( **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TridentNet(TridentBlock, [1, 2, 3], [3, 4, 6, 3], **kwargs)
    return model


net = tridentnet50()
for layer in list(net.Trident1)[0].layer:
    if isinstance(layer, nn.Conv2d):
        print("CONV1_weight:", layer.weight.size(), "CONV!_bias:", layer.bias)

for layer in list(net.Trident2)[0].layer:
    if isinstance(layer, nn.Conv2d):
        print("CONV2_weight:", layer.weight.size(), "CONV2_bias:", layer.bias)

