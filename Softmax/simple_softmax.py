import torch
from torch import nn
from torch.nn import init
import numpy as np
from collections import OrderedDict
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 定义初始化模型
num_inputs = 784
num_outputs = 10
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))
        return y


net = LinearNet(num_inputs, num_outputs)
net = nn.Sequential(
    OrderedDict([
        ('flatten', d2l.FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal(net.linear.weight, mean = 0, std = 0.01)
init.constant_(net.linear.bias, val = 0)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

