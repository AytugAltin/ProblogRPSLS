
import time

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from examples.RPSLS.RPSLS.helpers import format_string
from examples.RPSLS.RPSLS.normal_nets import *
from logger import Logger
import numpy as np
from torchvision import datasets, transforms
from params import *

def test_RPSLS(test_dataset):
    confusion = np.zeros((3, 3), dtype=np.uint32)  # First index actual, second index predicted
    correct = 0
    n = 0
    N = len(test_dataset)
    for d, l in test_dataset:
        d = Variable(d.unsqueeze(0))
        outputs = net.forward(d)
        _, out = torch.max(outputs.data, 1)
        c = int(out.squeeze())
        confusion[l, c] += 1
        if c == l:
            correct += 1
        n += 1
    acc = correct / n
    print(confusion)
    F1 = 0
    for nr in range(3):
        TP = confusion[nr, nr]
        FP = sum(confusion[:, nr]) - TP
        FN = sum(confusion[nr, :]) - TP
        F1 += 2 * TP / (2 * TP + FP + FN) * (FN + TP) / N
    print('F1: ', F1)
    print('Accuracy: ', acc)
    return F1,acc

class RPSLS_DataSet(Dataset):
    def __init__(self, dataset, examples):
        self.data = list()
        self.dataset = dataset
        with open(examples) as f:
            for line in f:
                line = format_string(line)
                line = line.strip().split(' ')
                self.data.append(tuple([int(i) for i in line]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        i1, i2, l = self.data[index]
        temp = torch.cat((self.dataset[i1][0], self.dataset[i2][0]), 1)
        return temp, l


print('Running RPSLS Baseline neural network without logic: ')
"""Choose Net"""

net = Net()
#net = DeepNet()
#net = Net2()

lr = 0.001
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=0.00001)
criterion = nn.CrossEntropyLoss()

print(lr)


transformations = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5), (0.5)),
    transforms.Grayscale(num_output_channels=1)
])
dataset = datasets.ImageFolder(
    root='../../../data/RPSLS/rock-paper-scissors-lizard-spock',
    transform = transformations)

train_dataset = RPSLS_DataSet(dataset, 'train_data.txt')
test_dataset = RPSLS_DataSet(dataset, 'test_data.txt')
trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=5 ,
                                          shuffle=True,
                                          num_workers=1)

i = 1
test_period = TEST_ITER
log_period = LOG_ITER
running_loss = 0.0
losslog = Logger()
accuracylog = Logger()
# net.eval()

start = time.time()
test_time = 0
for epoch in range(EPOCHS):
    print('Epoch: ', epoch)
    for data in trainloader:
        iter_time = time.time()
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

        # if i % log_period == 0:
        #     print('Iteration: ', i * 2, '\tAverage Loss: ', running_loss / log_period )
        #     log.log('loss', i * 2, running_loss / log_period)
        #     running_loss = 0
        # if i % test_period == 0:
        #     log.log('F1', i * 2, test_RPS(test_dataset))
        # i += 1

        if i % log_period == 0:
            print('Iteration: ', i ,
                  '\tAverage Loss: ', running_loss / log_period ,
                      '\ttime: ', iter_time - start - test_time,
                      '\tTest-time: ', test_time)
            losslog.log('loss', i , running_loss / log_period)
            losslog.log('time', i , iter_time - start- test_time)

            running_loss = 0
        if i % test_period == 0:
            test_start = time.time()
            f1,acc = test_RPSLS(test_dataset)
            #log.log('F1', i , f1)
            accuracylog.log('Accuracy', i , acc)
            test_time = test_time + (time.time() - test_start)
        i += 1

        if i % WRITE_PERIOD == 0:
            test_start = time.time()
            losslog.write_to_file("RPS_BaseLine_loss")
            accuracylog.write_to_file("RPS_BaseLine_accuracy")
            test_time = test_time + (time.time() - test_start)
    #
    # lr = lr *
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    print('learningrate: ', lr)


