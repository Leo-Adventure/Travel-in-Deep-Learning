import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape[0], K.shape[1]
    output_h = X.shape[0] - h + 1
    output_w = X.shape[1] - w + 1
    Y = torch.zeros((output_h, output_w))
    for i in range(output_h):
        for j in range(output_w):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()

    return Y

class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, input):
        return corr2d(input, self.weight) + self.bias

# 边缘检测
X = torch.ones((6, 8))
X[:, 2:6] = 0

K = torch.tensor([[1, -1]])
print(X)
Y = corr2d(X, K)

net = Conv2D((1, 2))
print(net)
num_epochs = 20
lr = 0.01
for i in range(num_epochs):
    y_hat = net(X)
    ls = ((y_hat - Y)** 2).sum()
    ls.backward()

    net.weight.data -= lr * net.weight.grad
    net.bias.data -= lr * net.bias.grad

    net.weight.grad.fill_(0)
    net.bias.grad.fill_(0)
    if (i + 1) % 5 == 0:
        print("In epoch %d, item = %.3f" % (i+1, ls.item()))

print(net.weight.data)

def conv_2d_mul_in(X, K):
    res = corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, K.shape[0]):
        res += corr2d(X[i, :, :], K[i, :, :])
    return res

X = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

print(conv_2d_mul_in(X, K))

def conv_2d_mul_in_mul_out(X, K):
    return torch.stack([conv_2d_mul_in(X, k) for k in K])

K = torch.stack([K, K + 1, K + 2])
print(K)

print(conv_2d_mul_in_mul_out(X, K))

