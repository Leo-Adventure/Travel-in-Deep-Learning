import torch.utils.data as Data
import torch
from torch import nn
from torch.nn import init
import numpy as np
import torch.optim as optim
import random

class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    #formard 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

# 读取数据
batch_size = 10
# 组合训练数据的特征以及标签
dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(dataset, batch_size, shuffle = True)

# 使用 nn.Sequential 来搭建网络，网络层将按照在传入 Sequential 的顺序依次被添加到计算图中
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
)

# 查看模型所有的可学习参数
# for param in net.parameters():
#     print(param)

# 初始化模型参数
init.normal_(net[0].weight, mean = 0, std=0.01)
init.constant_(net[0].bias, val=0)

# 损失函数
loss = nn.MSELoss()

# 优化算法
optimizer = optim.SGD(net.parameters(), lr = 0.03)

num_epoches = 3
for epoch in range(1, num_epoches + 1):
    for X,y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1,1))
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' %(epoch, l.item()))

dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
