# Normalization Layers 正则化层，主要用于模型的加速
# Recurrent Layers ，主要用于一些文字识别当中，特定的网络结构
# Transformer Layers ，特定的网络结构中提出的，pytorch把它当作已经写好的一个层-神经网络层来进行实现
# Linear Layers , 线性层，用到的较多
# Dropout Layers ,训练过程中随机地把input（Tensor类型）中随机的一些元素变成0，变成0的概率是p，若是1，则百分百变成0，主要是为了防止过拟合，可以练手
# Sparse Layers ,主要用于自然语言处理，也是用于特定的网络结构中
# Distance Functions ,主要计算两个值之间的误差是通过什么样的方式去衡量的
# Loss Functions , Loss误差的大小是多少

# 线性层Linear
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data.dataloader import DataLoader

data5 = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                     transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(data5, batch_size=64)

class MyDate(nn.Module):
    def __init__(self):
        super().__init__()
        #bias是偏置，如果为True就加b，为False就不加b，默认为True
        self.linear1=Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output

mydate = MyDate()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    #可将图像排列成一行
    output = torch.flatten(imgs)
    print(output.shape)
    # output = torch.reshape(imgs,(1,1,1,-1))   #reshape用法比flatten更为高级
    output = mydate(output)

    print(output.shape)


