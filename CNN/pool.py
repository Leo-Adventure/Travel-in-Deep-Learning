import torch
from torch import nn

def pooling(X, pool_size, mode='max'):
    X = X.float()
    h = X.shape[0] - pool_size[0] + 1
    w = X.shape[1] - pool_size[1] + 1
    Y = torch.zeros(h, w)
    for i in range(h):
        for j in range(w):
            if mode == 'max':
                Y[i, j] = X[i:i+pool_size[0], j:j+pool_size[1]].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+pool_size[0], j:j+pool_size[1]].mean()

    return Y

X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pooling(X, (2, 2), mode='avg'))