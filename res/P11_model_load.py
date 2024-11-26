# 加载模型
import torch
import torchvision
from torch import nn
from P11_model_save import *

# 方式1->保存方式1，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式2->保存方式2   以字典的形式读取
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

# 陷阱
# 引用整个模型class或者添加头文件from P11_model_save import *
# class Tudui(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3,64,kernel_size=3)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         return output

tudui = torch.load("tudui_method3.pth")
print(tudui)
