import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class DataLoader:
    def __init__(self, ini_feature_size, batch_size=4):
        self.transform = transforms.Compose(
            [
                transforms.Resize(ini_feature_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
        self.trainset = torchvision.datasets.FashionMNIST('./data',
            download=True,
            train=True,
            transform=self.transform)

        self.testset = torchvision.datasets.FashionMNIST('./data',
            download=True,
            train=False,
            transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=True)
        self.classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    def getTrainData(self):
        return self.trainloader
    
    def getTestData(self):
        return self.testloader






