import torch
from torch import nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def nin_blk(in_channel, out_channel, kernel_size, stride, padding):
    net = nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1),
        nn.ReLU()
    )
    return net

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, input):
        return F.avg_pool2d(input, kernel_size = input.size()[2:])

net = nn.Sequential(
    nin_blk(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_blk(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_blk(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    nin_blk(384, 10, kernel_size=3, stride=1, padding=1),
    GlobalAvgPool2d(),
    d2l.FlattenLayer()

)
# print(net)
X = torch.rand(1, 1, 226, 226)
for name, blk in net.named_children():
    X = blk(X)
    print(name, 'output shape: ', X.shape)