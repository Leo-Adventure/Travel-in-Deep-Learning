import torch
from torch import nn

# 不含有模型参数的神经层
class CenterLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenterLayer, self).__init__(**kwargs)
    def forward(self, X):
        print("X_mean = ", X.mean())
        return X - X.mean()



net = CenterLayer()
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(X)
y_hat = net(X)
print(y_hat)

net = nn.Sequential(nn.Linear(8, 128), CenterLayer())
print(net)

# 带有参数的模型
class MyDense(nn.Module):
    def __init__(self, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
    def forward(self, input):
        for i in range(len(self.params)):
                input = torch.mm(input, self.params[i])
        return input

net = MyDense()
print(net)

class MyDictDense(nn.Module):
    def __init__(self, **kwargs):
        super(MyDictDense, self).__init__(**kwargs)
        self.params = nn.ParameterDict({
            'Linear1' : nn.Parameter(torch.randn(4, 4)),
            'Linear2' : nn.Parameter(torch.randn(4, 1)),
        })
        self.params.update({'Linear3': nn.Parameter(torch.randn(4, 2))})

    def forward(self, input, choice='Linear1'):
        return torch.mm(input, self.params[choice])

net = MyDictDense()
print(net)