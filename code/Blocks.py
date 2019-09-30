import torch.nn as nn


# after conv calculate get a same size image
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class TridentBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilate=1, downsample=None):
        super(TridentBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, dilation=dilate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.layer = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.downsample]

    @classmethod
    def share_weight(cls, block1, block2, no_bias=True):
        # block1共享block2的权重（将block2的权重复制给block1）
        if not isinstance(block1, TridentBlock):
            raise Exception("wrongBlockClass")
        if not isinstance(block2, TridentBlock):
            raise Exception("wrongBlockClass")
        if no_bias:
            block1.conv1.weight = block2.conv1.weight
            block1.conv1.bias = None
            block1.bn1.weight = block2.bn1.weight
            block1.bn1.bias = block2.bn1.bias
            block1.conv2.weight = block2.conv2.weight
            block1.conv2.bias = None
            block1.bn2.weight = block2.bn2.weight
            block1.bn2.bias = block2.bn2.bias
            block1.conv3.weight = block2.conv3.weight
            block1.conv3.bias = None
            block1.bn3.weight = block2.bn3.weight
            block1.bn3.bias = block2.bn3.bias
            if block1.downsample is not None:
                list(block1.downsample)[0].weight = list(block2.downsample)[0].weight
                list(block1.downsample)[0].bias = None
                list(block1.downsample)[1].weight = list(block2.downsample)[1].weight
                list(block1.downsample)[1].bias = list(block2.downsample)[1].bias
        else:
            block1.conv1.weight = block2.conv1.weight
            block1.conv1.bias = block2.conv1.bias
            block1.bn1.weight = block2.bn1.weight
            block1.bn1.bias = block2.bn1.bias
            block1.conv2.weight = block2.conv2.weight
            block1.conv2.bias = block2.conv2.bias
            block1.bn2.weight = block2.bn2.weight
            block1.bn2.bias = block2.bn2.bias
            block1.conv3.weight = block2.conv3.weight
            block1.conv3.bias = block2.conv3.bias
            block1.bn3.weight = block2.bn3.weight
            block1.bn3.bias = block2.bn3.bias
            if block1.downsample is not None:
                list(block1.downsample)[0].weight = list(block2.downsample)[0].weight
                list(block1.downsample)[0].bias = list(block2.downsample)[0].bias
                list(block1.downsample)[1].weight = list(block2.downsample)[1].weight
                list(block1.downsample)[1].bias = list(block2.downsample)[1].bias

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

