# 非线性激活-最后的效果图片变模糊
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

data3 = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(data3, batch_size=64)


class MyDate(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

mydata = MyDate()

writer = SummaryWriter("../logs_ReLU")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = mydata(imgs)
    writer.add_images("output", output, global_step=step)

    step += 1

writer.close()
