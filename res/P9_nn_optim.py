# 优化器-optim  lr-学习速率

import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten
from torch.utils.data.dataloader import DataLoader

data8 = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(data8, batch_size=64)


class MyDate(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.seq(x)
        return x


mydata = MyDate()
# 损失函数loss
cross_loss = nn.CrossEntropyLoss()
# 优化器
optim = torch.optim.SGD(mydata.parameters(), lr=0.001)
for i in range(20):
    sum_loss_cross = 0.0
    for data in dataloader:
        imgs, targets = data
        output = mydata(imgs)
        result_cross = cross_loss(output, targets)
        optim.zero_grad()
        # 反向传播
        result_cross.backward()
        optim.step()
        sum_loss_cross = sum_loss_cross + result_cross
    print(sum_loss_cross)
