import torch
from torch import nn
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def vgg_stk(num_conv, in_channel, out_channel):
    blk = []
    for i in range(num_conv):
        if i == 0:
            blk.append(nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1))
        else:
            blk.append(nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*blk)


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_features = 512 * 7 * 7 # c * w * h
fc_hidden_units = 4096 # 任意

def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i , (num_conv, in_channel, out_channel) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_stk(num_conv, in_channel, out_channel))

    net.add_module("fc", nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(fc_features, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, fc_hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(fc_hidden_units, 10)
    ))
    return net

net = vgg(conv_arch, fc_features, fc_hidden_units)
print(net)
X = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    X = blk(X)
    print(name, "output shape = ", X.shape)

