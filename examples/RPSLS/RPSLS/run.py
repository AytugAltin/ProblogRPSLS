from torchvision import models
from examples.RPSLS.RPSLS.logic_nets import *
from train import train_model_new
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from examples.RPSLS.RPSLS.helpers import *
import torch
from params import *


print('Running RPSLS PROBLOG: ')

queries = load('train_data.txt')
test_queries = load('test_data.txt')

with open('model.pl') as f:
    problog_string = f.read()

network = RPSLS_Net(N=5)

learning_rate = 0.001
print(learning_rate)

net = Network(network, 'rpsls_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model_new(model, queries, nr_epochs=EPOCHS, optimizer=optimizer,test_iter=TEST_ITER,
            test = lambda x: Model.accuracy(x, test_queries),log_iter=LOG_ITER)




