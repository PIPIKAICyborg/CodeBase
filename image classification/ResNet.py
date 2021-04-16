import numpy as np
import matplotlib.pyplot as plt

import torch 
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

class InputBlock(nn.Module):
    def __init__(self, in_channel):
        super(InputBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, input_channel, output_channel, strides,expansion):
        super(BasicBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.strides = strides
        # 不需要偏执的原因是，如果后面接BN层，那么也就不需要偏置
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(output_channel),
        )
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if(self.strides != 1):
            residual = self.downsample(input)
        x += residual
        x = self.relu2(x)
        return x

class BottleBlock(nn.Module):
    def __init__(self, input_channel, output_channel, strides, expansion):
        super(BottleBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.strides = strides
        self.expansion = expansion
        # 不需要偏执的原因是，如果后面接BN层，那么也就不需要偏置
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(output_channel, output_channel*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_channel*self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
                nn.Conv2d(input_channel, output_channel*self.expansion, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(output_channel*self.expansion),
        )
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if(self.strides != 1 or self.input_channel != self.output_channel*self.expansion):
            residual = self.downsample(input)
        x += residual
        x = self.relu3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, blocks, n_classes=1000, input_channel=1, expansion=1):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.inputBlock = InputBlock(input_channel)
        
        self.layer1 = self.make_layer(64, 64, 1, blocks[0], block)
        self.layer2 = self.make_layer(64*self.expansion, 128, 2, blocks[1], block)
        self.layer3 = self.make_layer(128*self.expansion, 256, 2, blocks[2], block)
        self.layer4 = self.make_layer(256*self.expansion, 512, 2, blocks[3], block)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512*self.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    def make_layer(self, input_channel, output_channel, strides, blocks_num, block):
        layers = []
        for i in range(blocks_num):
            if(i == 0):
                layers.append(block(input_channel, output_channel, strides, 4))
            else:
                layers.append(block(output_channel*self.expansion, output_channel, 1, 4))
        return nn.Sequential(*layers)
    
    def forward(self, input):
        x = self.inputBlock(input)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

net = ResNet(BottleBlock, [3, 4, 6, 3], expansion=4)
x = torch.rand((1, 1, 224, 224))
net(x)
print("no problem!!!!")
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# writer.add_graph(net, images)
# writer.close()