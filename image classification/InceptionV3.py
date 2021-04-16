# InceptionV1/GoogleNet: 
# --------------------- efficient in number of parameters and memory cost
# InceptionV3 improvments:
# --------------------- 1.Factorizing Convolutions
# ---------------------     1.1.Smaller convolutions e.g. replace 5*5 filter with two 3*3 filter
# ---------------------     1.2.Asymmetric convolutions e.g. replace 3*3 with the 1*3 followed by a 3*1 filter 
# --------------------- 2.Auxiliary classifier is a small CNN inserted between layers during training(as a regularizer) 
# --------------------- 3.Grid size reduction, using pooling

import torch
import torch.nn as nn


class Basic_conv(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(Basic_conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return nn.ReLU(inplace=True)(x)

# BlockA:
# -------branch1: -->conv1*1(out_channel=64)
# -------branch2: -->conv1*1(out_channel=48)-->conv5*5(out_channel=64)
# -------branch3: -->conv1*1(out_channel=64)-->conv3*3(out_channel=96)-->conv3*3(out_channel=96)
# -------branch4: -->avgpool3*3(stride=1, padding=1)-->conv1*1(out_channel=32)
class InceptionBlockA(nn.Module):
    def __init__(self, in_channel, pool_feature):
        super(InceptionBlockA, self).__init__()
        self.branch1 = Basic_conv(in_channel, 64, kernel_size=1)

        self.branch2_1 = Basic_conv(in_channel, 48, kernel_size=1)
        self.branch2_2 = Basic_conv(48, 64, kernel_size=5, padding=2)

        self.branch3_1 = Basic_conv(in_channel, 64, kernel_size=1)
        self.branch3_2 = Basic_conv(64, 96, kernel_size=3, padding=1)
        self.branch3_3 = Basic_conv(96, 96, kernel_size=3, padding=1)

        self.branch4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = Basic_conv(in_channel, 32, kernel_size=1)
    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        branch3 = self.branch3_3(branch3)

        branch4 = self.branch4_1(x)
        branch4 = self.branch4_2(branch4)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# BlockB:
# -------branch1: -->conv3*3(out_channel=384, stride=2)
# -------branch2: -->conv1*1(out_channel=64)-->conv3*3(out_channel=96)-->conv3*3(out_channel=96, stride=2)
# -------branch3: -->avgpool3*3(stride=2)
class InceptionBlockB(nn.Module):
    def __init__(self, in_channel, pool_feature):
        super(InceptionBlockB, self).__init__()
        self.branch1 = Basic_conv(in_channel, 384, kernel_size=3, stride=2)

        self.branch2_1 = Basic_conv(in_channel, 64, kernel_size=1)
        self.branch2_2 = Basic_conv(64, 96, kernel_size=3, padding=1)
        self.branch2_3 = Basic_conv(96, 96, kernel_size=3, stride=2, padding=1)

        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2)
    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)

        branch3 = self.branch3(x)

        return torch.cat([branch1, branch2, branch3], dim=1)

# BlockC:(*c7 is the out_channel for con1*7 and conv7*1)
# -------branch1: -->conv1*1(out_channel=192)
# -------branch2: -->conv1*1(out_channel=c7)-->conv1*7(out_channel=c7, padding=(0,3))-->conv7*1(out_channel=192, padding=(3,0))
# -------branch3: -->conv1*1(out_channel=c7)-->conv7*1(out_channel=c7, padding=(3,0))-->conv1*7(out_channel=c7, padding=(0,3))-->conv7*1(out_channel=c7, padding=(3,0))-->conv1*7(out_channel=192, padding=(0,3))
# -------branch4: -->avgpool3*3(stride=1, padding=1)->conv1*1(out_channel=192)
class InceptionBlockC(nn.Module):
    def __init__(self, in_channel, pool_feature， channels_7*7):
        super(InceptionBlockC, self).__init__()
        self.branch1 = Basic_conv(in_channel, 192, kernel_size=1)

        c7 = channels_7*7
        self.branch2_1 = Basic_conv(in_channel, c7, kernel_size=1)
        self.branch2_2 = Basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_3 = Basic_conv(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch3_1 = Basic_conv(in_channel, c7, kernel_size=1)
        self.branch3_2 = Basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_3 = Basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_4 = Basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_5 = Basic_conv(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = Basic_conv(in_channel, 192, kernel_size=1)
    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        branch3 = self.branch3_3(branch3)
        branch3 = self.branch3_4(branch3)
        branch3 = self.branch3_5(branch3)

        branch4 = self.branch4_1(x)
        branch4 = self.branch4_2(branch4)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)

# Not finished
# BlockD:(*c7 is the out_channel for con1*7 and conv7*1)
# -------branch1: -->conv1*1(out_channel=192)
# -------branch2: -->conv1*1(out_channel=c7)-->conv1*7(out_channel=c7, padding=(0,3))-->conv7*1(out_channel=192, padding=(3,0))
# -------branch3: -->conv1*1(out_channel=c7)-->conv7*1(out_channel=c7, padding=(3,0))-->conv1*7(out_channel=c7, padding=(0,3))-->conv7*1(out_channel=c7, padding=(3,0))-->conv1*7(out_channel=192, padding=(0,3))
# -------branch4: -->avgpool3*3(stride=1, padding=1)->conv1*1(out_channel=192)
class InceptionBlockD(nn.Module):
    def __init__(self, in_channel, pool_feature， channels_7*7):
        super(InceptionBlockC, self).__init__()
        self.branch1 = Basic_conv(in_channel, 192, kernel_size=1)

        c7 = channels_7*7
        self.branch2_1 = Basic_conv(in_channel, c7, kernel_size=1)
        self.branch2_2 = Basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch2_3 = Basic_conv(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch3_1 = Basic_conv(in_channel, c7, kernel_size=1)
        self.branch3_2 = Basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_3 = Basic_conv(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch3_4 = Basic_conv(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch3_5 = Basic_conv(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_2 = Basic_conv(in_channel, 192, kernel_size=1)
    def forward(self, x):
        branch1 = self.branch1(x)

        branch2 = self.branch2_1(x)
        branch2 = self.branch2_2(branch2)
        branch2 = self.branch2_3(branch2)

        branch3 = self.branch3_1(x)
        branch3 = self.branch3_2(branch3)
        branch3 = self.branch3_3(branch3)
        branch3 = self.branch3_4(branch3)
        branch3 = self.branch3_5(branch3)

        branch4 = self.branch4_1(x)
        branch4 = self.branch4_2(branch4)

        return torch.cat([branch1, branch2, branch3, branch4], dim=1)



class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()