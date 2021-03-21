from torchvision import models
from examples.RPSLS.RPS.logic_nets import RPS_Net
from train import train_model_new
from data_loader import load
from model import Model
from optimizer import Optimizer
from network import Network
from examples.RPSLS.RPS.helpers import *
import torch
from params import *


print('Running RPS PROBLOG: ')

queries = load('train_data.txt')
test_queries = load('test_data.txt')

with open('model.pl') as f:
    problog_string = f.read()

network = RPS_Net(N=3)

learning_rate = 0.0001

net = Network(network, 'rps_net', neural_predicate)
net.optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
model = Model(problog_string, [net], caching=False)
optimizer = Optimizer(model, 2)

train_model_new(model, queries, nr_epochs=EPOCHS, optimizer=optimizer,test_iter=TEST_ITER,
            test = lambda x: Model.accuracy(x, test_queries),log_iter=LOG_ITER)




