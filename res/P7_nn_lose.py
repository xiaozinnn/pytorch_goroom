# loss 损失
# 1、计算我们输出和目标之间的差距；2、为我们更新输出提供一定的依据（方向传播）
# x:1,2,3  y:1,2,5    loss = (|1-1|+|2-2|+|3-5|)/3=0.6 loss值越小越好

import torch
import torchvision
from torch.nn import L1Loss
from torch import nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss()
result = loss(input, target)

# 计算的是差的平方（0+0+2*2）/3 = 1.333
loss_mse = nn.MSELoss()
result_mse = loss_mse(input, target)
print(result)
print(result_mse)


x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, [1, 3])
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
