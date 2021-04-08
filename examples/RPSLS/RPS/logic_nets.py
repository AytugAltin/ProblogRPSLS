import pickle
from zipfile import ZipFile

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from network import Network
from examples.NIPS.MNIST.mnist import MNIST_Net, neural_predicate


class RPS_Net(nn.Module):
    def __init__(self, N=3):
        super(RPS_Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,  6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 9 * 9, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, N),
            nn.Softmax(1))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 9 * 9)
        x = self.classifier(x)
        return x

    def save_state(self, location):
        with ZipFile(location,'w') as zipf:
            with zipf.open('parameters','w') as f:
                pickle.dump(self.parameters,f)
            for n in self.networks:
                with zipf.open(n,'w') as f:
                    self.networks[n].save(f)
