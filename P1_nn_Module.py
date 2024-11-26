# 深度网络
import torch
from torch import nn


class Hpnn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


hpnn = Hpnn()
x = torch.tensor(1.0)
output = hpnn(x)
print(output)
