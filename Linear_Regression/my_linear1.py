import torch
import numpy as np
import random

# 随机读取数据
def data_iter(batch_size, features, labels):
    num_example = len(features)
    indices = list(range(num_example))
    random.shuffle(indices)
    for i in range(0, num_example, batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size, num_example)])
        yield features.index_select(0, j), labels.index_select(0, j)

# 定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b


# 损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2/ 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= param.grad * lr / batch_size

# 生成数据集
num_inputs = 2
num_example = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_example, num_inputs, dtype = torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
# 噪声
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)

# 读取数据
batch_size = 10

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad = True)
b.requires_grad_(requires_grad = True)

# 训练模型
num_epoch = 3
lr = 0.03
net = linreg
loss = squared_loss


for epoch in range(num_epoch):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        l.backward()
        # 梯度下降
        sgd([w, b], lr, batch_size)
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % (epoch+ 1, train_l.mean().item()))

print(true_w, "\n", w)

print(true_b, '\n', b)