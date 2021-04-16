import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, kernel_size=5, padding=0, bias=True)
        self.sig1 = nn.Sigmoid()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=True)
        self.sig2 = nn.Sigmoid()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dense1 = nn.Linear(256, 120)
        self.dense2 = nn.Linear(120, 84)
        self.dense3 = nn.Linear(84, num_classes)

    def forward(self, input):
        x = self.conv1(input)
        x = self.sig1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.sig2(x)
        x = self.maxpool2(x)

        x = x.view(-1, 4*4*16)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model = LeNet(1, 10)
x = torch.rand((1, 1, 28, 28))
model(x)