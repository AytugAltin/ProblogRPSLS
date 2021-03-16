import random
import torchvision
from torchvision import datasets, transforms
from data.RPSLS.types import *

transformations = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(
    root='../../../data/RPSLS/rock-paper-scissors-lizard-spock',
    transform = transformations
)

def generate_data(dataset):
    examples = list()
    i = list(range(len(dataset)))
    random.shuffle(i)
    i = iter(i)
    while True:
        try:
            examples.append(next_example(i,dataset))
        except StopIteration:
            break

def next_example(i,dataset):
    x1, x2 = next(i), next(i)
    y = winner(x1, x2,dataset)
    return x1, x2, y

generate_data(dataset)