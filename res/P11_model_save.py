# 保存模型
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)
# print(vgg16)

# 保存方式1:模型结构+模型参数
# torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2:模型参数  以字典的形式保存（官方推荐）
# torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3)

    def forward(self, input):
        output = self.conv1(input)
        return output

tudui = Tudui()

#保存模型
# torch.save(tudui,"tudui_method3.pth")