import torch
import torch.nn as nn

class BottleBlock(nn.Module):
    def __init__(self, input_channel, growth_rate, bn_size, drop_rate):
        super(BottleBlock, self).__init__()
        self.input_channel = input_channel
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        # 不需要偏执的原因是，如果后面接BN层，那么也就不需要偏置
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, bn_size*growth_rate, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, input):
        x = self.bn1(input)
        x = self.relu1(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = nn.Dropout(p=self.drop_rate)(x)
        return torch.cat([x, input], dim=1)

class InputBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(InputBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, block_num, in_channels, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()
        self.net = []
        for i in range(block_num):
            self.net.append(BottleBlock(in_channels+i*growth_rate, growth_rate, bn_size, drop_rate))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, X):
        X = self.net(X)
        print(X.shape)
        return X

class TransitionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        print(x.shape)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)
        print(x.shape)
        return x

class DenseNet(nn.Module):
    def __init__(
        self, 
        blocks, 
        in_channel, 
        growth_rate, 
        bn_size, 
        feature_channel, 
        drop_rate,
        num_classes):
        super(DenseNet, self).__init__()
        self.feature_extract = InputBlock(in_channel, feature_channel)

        self.DenseBlocks = []
        num_features = feature_channel
        for i, num_layers in enumerate(blocks):
            self.DenseBlocks.append(DenseBlock(num_layers, num_features, growth_rate, bn_size, drop_rate))
            num_features += num_layers * growth_rate
            if(i != len(blocks)-1):
                self.DenseBlocks.append(TransitionBlock(num_features, num_features//2))
                num_features = num_features // 2
        self.DenseBlocks.append(nn.BatchNorm2d(num_features))
        self.DenseBlocks = nn.Sequential(*self.DenseBlocks)

        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        features = self.feature_extract(x)
        features = self.DenseBlocks(features)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

dense = DenseNet([6, 12, 24, 16],3, 2, 4, 64, 0.2, 10)
x = torch.rand([4, 3, 224, 224])
x = dense(x)
print(x.shape)