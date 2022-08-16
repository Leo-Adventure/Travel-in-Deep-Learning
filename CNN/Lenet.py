import torch
from torch import nn

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