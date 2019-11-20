import torch.nn as nn
from Module.Blocks import conv1x1
from Module.Blocks import TridentBlock


class TridentHead(nn.Module):

    def __init__(self, block, dilate, layer_num, zero_init_residual=False):
        super(TridentHead, self).__init__()
        """
        TridentHeads replace the last layer of the ResNet
        """
        self.inplanes = [512, 512, 512]
        self.Trident1 = self._make_layer(block, 0, 512, layer_num, dilation=dilate[0], stride=2)
        self.Trident2 = self._make_layer(block, 1, 512, layer_num, dilation=dilate[1], stride=2)
        self.Trident3 = self._make_layer(block, 2, 512, layer_num, dilation=dilate[2], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, TridentBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, branch, planes, blocks, dilation=1, stride=1):
        downsample = None
        if stride != 1 or self.inplanes[branch] != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes[branch], planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes[branch], planes, stride, dilation, downsample))
        self.inplanes[branch] = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes[branch], planes, dilate=dilation))

        return nn.Sequential(*layers)

    def share_weight(self, name, no_bias=True):
        """share weight between 3 Trident branch"""
        if name not in ['Trident1', 'Trident2', 'Trident3']:
            raise Exception("WrongName")
        if no_bias:
            if name == 'Trident1':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T2, T1)
                    TridentBlock.share_weight(T3, T1)
            if name == 'Trident2':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T1, T2)
                    TridentBlock.share_weight(T3, T2)
            if name == 'Trident3':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T1, T3)
                    TridentBlock.share_weight(T2, T3)
        else:
            if name == 'Trident1':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T2, T1, no_bias=False)
                    TridentBlock.share_weight(T3, T1, no_bias=False)
            if name == 'Trident2':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T1, T2, no_bias=False)
                    TridentBlock.share_weight(T3, T2, no_bias=False)
            if name == 'Trident3':
                for (T1, T2, T3) in zip(list(self.Trident1), list(self.Trident2), list(self.Trident3)):
                    TridentBlock.share_weight(T1, T3, no_bias=False)
                    TridentBlock.share_weight(T2, T3, no_bias=False)

    @staticmethod
    def layer_share_weight(net, name, no_bias=True):
        """share weight between 3 Trident branch"""
        if name not in ['Trident1', 'Trident2', 'Trident3']:
            raise Exception("WrongName")
        if no_bias:
            if name == 'Trident1':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T2, T1)
                    TridentBlock.share_weight(T3, T1)
            if name == 'Trident2':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T1, T2)
                    TridentBlock.share_weight(T3, T2)
            if name == 'Trident3':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T1, T3)
                    TridentBlock.share_weight(T2, T3)
        else:
            if name == 'Trident1':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T2, T1, no_bias=False)
                    TridentBlock.share_weight(T3, T1, no_bias=False)
            if name == 'Trident2':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T1, T2, no_bias=False)
                    TridentBlock.share_weight(T3, T2, no_bias=False)
            if name == 'Trident3':
                for (T1, T2, T3) in zip(list(net.Trident1), list(net.Trident2), list(net.Trident3)):
                    TridentBlock.share_weight(T1, T3, no_bias=False)
                    TridentBlock.share_weight(T2, T3, no_bias=False)

    def forward(self, x):
        trident1 = self.Trident1(x)
        trident2 = self.Trident2(x)
        trident3 = self.Trident3(x)

        return trident1, trident2, trident3
