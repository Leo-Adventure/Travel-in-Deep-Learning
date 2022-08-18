import time
import torch
from torch import nn, optim

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__(**kwargs)
        self.Conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5,5)),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )
    def forward(self, input):
        feature = self.Conv(input)
        output = self.fc(feature.view(input.shape[0], -1))
        return output

net = LeNet()
print(net)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def evaluate_accuracy(net, data_iter, device=None):
    acc_sum = 0.0
    n = 0
    if device != None and isinstance(net, nn.Module):
        device = list(net.parameters())[0].device
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum()
            net.train()

        n += y.shape[0]
    return acc_sum/n

def train_ch5(net, train_iter, test_iter, num_epochs, optimizer):
    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    train_ls_sum = 0.0
    train_acc_sum = 0.0
    n = 0
    batch_count = 0


    for epoch in range(num_epochs):
        start = time.time()
        print(start)
        cnt = 0
        for X, y in train_iter:
            print(cnt)
            cnt += 1
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls_sum += l.cpu().item()
            train_acc_sum += evaluate_accuracy(net, train_iter, device=device)
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(net, test_iter, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_ls_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(net.parameters(), lr = lr)
train_ch5(net, train_iter, test_iter, num_epochs, optimizer)
