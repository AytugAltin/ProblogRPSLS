import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, outcomes=3):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 16 * 22 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, outcomes),
            nn.Softmax(1))


    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1 * 16 * 22 * 9)
        x = self.classifier(x)
        return x

class DeepNet(nn.Module):
    def __init__(self, outcomes=3):
        super(DeepNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(16, 25, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 25 * 22 * 9, 300),
            nn.ReLU(),
            nn.Linear(300, 120),
            nn.ReLU(),
            nn.Linear(120, outcomes),
            nn.Softmax(1))


        print("DeepNet is being use")

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1 * 25 * 22 * 9)
        x = self.classifier(x)
        return x



class SimpleNet(nn.Module):
    def __init__(self, outcomes=3):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1 * 16 * 22 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, outcomes),
            nn.Softmax(1))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 1 * 16 * 22 * 9)
        x = self.classifier(x)
        return x