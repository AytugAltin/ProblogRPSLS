import random
from torchvision import datasets, transforms
from examples.RPSLS.RPS.helpers import *

TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

transformations = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(
    root='../../../data/RPSLS/rock-paper-scissors/raw',
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


    train_dataset,test_dataset = split_examples(examples,TRAIN_RATIO,TEST_RATIO)

    write_dataset(train_dataset, 'train_data.txt', 'train')
    write_dataset(test_dataset, 'test_data.txt', 'test')


def write_dataset(examples,filename,dataset_name):
    with open(filename, 'w') as f:
        for example in examples:
            args = tuple('{}({})'.format(dataset_name, e) for e in example[:-1])
            f.write('rps({},{},{}).\n'.format(*args, example[-1]))


def next_example(i,dataset):
    x1, x2 = next(i), next(i)
    y = winner(x1, x2,dataset)
    return x1, x2, y

generate_data(dataset)