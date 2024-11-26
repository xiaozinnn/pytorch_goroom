# 最大池化层-最终的效果是特征点比较明显
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

data2 = torchvision.datasets.CIFAR10("./data",train=False,download=True,
                                     transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(data2,batch_size=64)


class Mydate(nn.Module):
    def __init__(self):
        super().__init__()
        #ceil_mode=True的话就取边缘的值，false不取边缘值就是不保留
        self.maxpool1 = MaxPool2d(kernel_size=3,ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

mydate = Mydate()

writer = SummaryWriter("./logs_maxpool")
temp = 0
for data in dataloader:
    imgs,targets = data
    output = mydate(imgs)
    writer.add_images("inout",imgs,temp)
    writer.add_images("output",output,temp)
    temp+=1

writer.close()
