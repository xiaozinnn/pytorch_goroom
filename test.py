# 测试模型
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Linear, ReLU
from torch.nn.modules.flatten import Flatten

image_path = "./images/002.png"
image = Image.open(image_path)
image = image.convert('RGB')

trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                        torchvision.transforms.ToTensor()])

image = trans(image)


# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            ReLU(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


model = torch.load("./tudui_100.pth")
image = torch.reshape(image, (1, 3, 32, 32))
# 需要给输入转换为cuda，因为此时模型就是cuda类型的
image = image.cuda()
# 以验证模型开启模型
model.eval()
with torch.no_grad():
    output1 = model(image)

print(output1.argmax(1).item())
