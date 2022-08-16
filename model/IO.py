import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(3, 2)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, input):
        output = self.relu(self.hidden(input))
        output = self.output(output)
        return output

net = MLP()
print(net.state_dict())

input = torch.randn(4, 3)
output = net(input)
torch.save(net.state_dict(), "state.pt")
print("output = ", output)

net2 = MLP()
net2.load_state_dict(torch.load("state.pt"))

output2 = net(input)
print("output2 = ", output2)

print(torch.cuda.is_available())
