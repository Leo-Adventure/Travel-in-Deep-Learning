import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

for name, param in net.named_parameters():
    print(name, param.size())

for name, param in net.named_parameters():
    print(param.data)

def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, weight in net.named_parameters():
    if 'weight' in name:
        init_weight_(weight)
        print(name, weight.data)

linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear)


for name, param in net.named_parameters():
    init.constant_(param,val=3.0)
    print(param.data)

X = torch.ones(1, 1)
print(X)
y_hat = net(X)

print(y_hat.sum())
y_hat.backward()
print(net[0].weight.grad)
