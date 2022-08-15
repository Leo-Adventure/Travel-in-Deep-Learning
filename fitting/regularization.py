import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 定义数据集
num_train, num_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
features = torch.randn((num_train + num_test, num_inputs))
labels = torch.matmul(features, true_w) +true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:num_train, :], features[num_train:, :]
train_labels = labels[:num_train]
test_labels = labels[num_train:]

# 初始化模型参数
def init():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义范数惩罚项
def l2_panelty(w):
    return (w**2).sum() / 2


# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)
w, b = init()
def fit_and_plot(lamb):

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X, w, b), y) + lamb * l2_panelty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            d2l.sgd([w, b], lr, batch_size)

            train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
            test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
        print('L2 norm of w:', w.norm().item())
        print("loss = ", test_ls[-1])

fit_and_plot(lamb=3)
print("##################################################################")

def fit_pt(lamb):
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, 0, 1)
    nn.init.normal_(net.bias, 0, 1)
    optimizer_w = torch.optim.SGD([net.weight], weight_decay=lamb, lr = lr)
    optimizer_b = torch.optim.SGD([net.bias], weight_decay=lamb, lr = lr)

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()
            optimizer_b.step()
            optimizer_w.step()

        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
        print('L2 norm of w:', net.weight.data.norm().item())
        print("loss = ", test_ls[-1])

fit_pt(lamb=0)








