# 现有的vgg16模型-分类模型    最新版本将pretrained替换为了weight,如下2、3
# vgg16_false = torchvision.models.vgg16(weights =None)
# vgg16_true = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
import torch
import torchvision
from torch import nn

# 下载vgg16的训练集，但是图片太大，无法下载
# data10 = torchvision.datasets.ImageNet("./data_image_net",split='train',download=True,
#                                        transform=torchvision.transforms.ToTensor())

# 拉取vgg16模型 路径在C:\Users\代宁\.cache\torch\checkpoints
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)

data10 = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                      transform=torchvision.transforms.ToTensor())

# 对vgg16_true（现有的分类模型，加上了预训练的参数版本）添加网络模型
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))
print(vgg16_true)

# 对vgg16_false(现有的分类模型，未加上预训练的参数版本,即使用的是初始化的参数，是没有经过训练的)修改网络模型
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
