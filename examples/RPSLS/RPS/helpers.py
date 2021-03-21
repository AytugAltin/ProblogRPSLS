import torchvision.transforms as transforms
from torch.autograd import Variable
import os
from torchvision import datasets
from train import interrupt, train
from logger import Logger
import time
import random
import csv

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)

def winner(x1,x2,dataset):
    (_, c1), (_, c2)  = dataset[x1], dataset[x2]
    classes = dataset.classes

    if c1 == c2:
        return 0 #SAME CLASS

    c1 = classes[c1]
    c2 = classes[c2]

    if c1 == "paper":
        if c2 == "rock":
            return 1
        if c2 == "scissors":
            return 2

    if c1 == "scissors":
        if c2 == "paper":
            return 1
        if c2 == "rock":
            return 2

    if c1 == "rock":
        if c2 == "scissors":
            return 1
        if c2 == "paper":
            return 2

    raise Exception("label not recognized")

def split_examples(examples,TRAIN_RATIO,TEST_RATIO):
    size = round(len(examples) * TRAIN_RATIO)
    train_data = examples[:size]
    test_data = examples[size:]
    return train_data,test_data


def neural_predicate(network, i):
    i = int(i.args[0])
    d, l = dataset[i]
    d = Variable(d.unsqueeze(0))
    output = network.net(d)
    return output.squeeze(0)


transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
])

dataset = datasets.ImageFolder(root='../../../data/RPSLS/rock-paper-scissors',transform = transformations)

def format_string(string):
    string = string.replace('rps', '')
    string = string.replace('test', '')
    string = string.replace('train', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    string = string.replace('.', '')
    string = string.replace(',', ' ')
    return string

def write(file,first_coll,second_coll):
    return 0

