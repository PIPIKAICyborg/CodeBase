import torch
import torch.nn as nn

class Basic_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, strides, padding):
        super(Basic_Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, bias=False, stride=strides,padding=padding)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Inception(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, out_channel3, out_channel4):
        super(Inception, self).__init__()

        self.branch1 = Basic_Conv(in_channel, out_channel1, 1, 1, 0)

        self.branch2 = nn.Sequential(
            Basic_Conv(in_channel, out_channel2[0], 1, 1, 0),
            Basic_Conv(out_channel2[0], out_channel2[1], 3, 1, 1)
        )

        self.branch3 = nn.Sequential(
            Basic_Conv(in_channel, out_channel3[0], 1, 1, 0),
            Basic_Conv(out_channel3[0], out_channel3[1], 5, 1, 2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1,ceil_mode=True),
            Basic_Conv(in_channel, out_channel4, kernel_size=1, strides=1, padding=0)
        )
    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        x4 = self.branch4(inputs)

        return torch.cat([x1, x2, x3, x4], 1)

class GoogleNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(GoogleNet, self).__init__()
        self.conv1 = Basic_Conv(input_channel, 64, kernel_size=7, strides=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = Basic_Conv(64, 64, kernel_size=1, strides=1, padding=0)
        self.conv3 = Basic_Conv(64, 192, kernel_size=3, strides=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, [96, 128], [16, 32], 32)
        self.inception3b = Inception(256, 128, [128, 192], [32, 96], 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, [96, 208], [16, 48], 64)
        self.inception4b = Inception(512, 160, [112, 224], [24, 64], 64)
        self.inception4c = Inception(512, 128, [128, 256], [24, 64], 64)
        self.inception4d = Inception(512, 112, [144, 288], [32, 64], 64)
        self.inception4e = Inception(528, 256, [160, 320], [32, 128], 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, [160, 320], [32, 128], 128)
        self.inception5b = Inception(832, 384, [192, 384], [48, 128], 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x

net = GoogleNet(3, 1000)
x = torch.rand((1, 3, 224, 224))
net(x)