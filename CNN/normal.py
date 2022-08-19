import torch
from torch import nn
import sys
sys.path.append("")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def batch_norm(is_training, X, moving_mean, moving_var, eps, momentum, gamma, beta):
    if not is_training:
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert (X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X-mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
            var = ((X-mean) **2).mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_hat = (X - mean)/torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta
    return Y, moving_mean, moving_var

class Batch_Norm(nn.Module):
    def __init__(self, num_feature, num_dims):
        super(Batch_Norm, self).__init__()
        if num_dims == 2:
            shape = (1, num_feature)
        else:
            shape = (1, num_feature, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self, input):
        if self.moving_mean.device != input.device:
            self.moving_mean = self.moving_mean.to(input.device)
            self.moving_var = self.moving_var.to(input.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, X=input, moving_mean=self.moving_mean, moving_var=self.moving_var,
                                                gamma=self.gamma, beta=self.beta, momentum=0.9, eps=1e-5)
        return Y

# LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    Batch_Norm(6, 4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    Batch_Norm(16, 4),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    d2l.FlattenLayer(),
    nn.Linear(16*4*4, 120),
    Batch_Norm(120, 2),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    Batch_Norm(84, 2),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)