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

class Net2(nn.Module):
    def __init__(self, outcomes=3):
        super(Net2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 12, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(12 * 22 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 400),
            nn.ReLU(),
            nn.Linear(400, 84),
            nn.ReLU(),
            nn.Linear(84, outcomes),
            nn.Softmax(1))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 12*22*9)
        x = self.classifier(x)
        return x


class SimpleNet(nn.Module):
    def __init__(self, outcomes=3):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        #self.conv3 = nn.Conv2d(in_channels=12,out_channels=24,kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*22*9, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.fc3 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=outcomes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2,stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2,stride=2)

        # x = self.conv3(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x,kernel_size=2,stride=2)

        x = x.reshape(-1,12*22*9)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        #x = F.softmax(x, dim=1)

        return x