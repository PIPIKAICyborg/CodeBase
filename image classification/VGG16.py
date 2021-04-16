import torch
import torch.nn as nn
import torch.optim as optim

import torchvision as torchvision
import torchvision.transforms as transforms

class VGG16(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(VGG16, self).__init__()
        self.conv_block1 = self.make_conv_maxpool_block(2, in_channel, 64)
        self.conv_block2 = self.make_conv_maxpool_block(2, 64, 128)
        self.conv_block3 = self.make_conv_maxpool_block(3, 128, 256)
        self.conv_block4 = self.make_conv_maxpool_block(3, 256, 512)
        self.conv_block5 = self.make_conv_maxpool_block(3, 512, 512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(1, -1)
        self.classfier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.classfier(x)

        return x
    
    def make_conv_maxpool_block(self, conv_num, in_channel, out_channel):
        layers = []

        for i in range(conv_num):
            if (i==0):
                layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
                layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

net = VGG16(1, 10)
x = torch.rand((1, 1, 224, 224))
net(x)