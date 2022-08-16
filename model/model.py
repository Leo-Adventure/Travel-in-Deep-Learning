import torch
from collections import OrderedDict
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
X = torch.rand(2, 784)
print(net)
print(net(X))

class MySequential(nn.Module):

    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input

print("#############################")
net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
X = torch.rand(2, 784)
print(net)
print(net(X))

class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__()
        self.constant_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    def forward(self, input):
        input = self.linear(input)
        input = nn.functional.relu(torch.mm(input, self.constant_weight.data) + 1)
        input = self.linear(input)
        while input.norm().item() > 1:
            input = input / 2
        if input.norm().item() < 0.8:
            input *= 10
        return input.sum()

class NestMLP(nn.Module):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(40, 30), nn.ReLU())
    def forward(self, input):
        return self.net(input)

net = nn.Sequential(NestMLP(), nn.Linear(30, 20), FancyMLP())
X = torch.rand(2, 40)
print(net)
print(net(X))


