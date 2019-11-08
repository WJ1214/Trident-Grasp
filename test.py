from TridentHead import TridentHead
from Blocks import TridentBlock
import torch.nn as nn


def tridentnet50( **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TridentHead(TridentBlock, [1, 2, 3], 3, **kwargs)
    return model


net = tridentnet50()
# TridentNet.layer_share_weight(net, 'Trident3')
