# 测试模型
import torch
import torchvision
from PIL import Image
from P12_model import Tudui

image_path = "./images/dog.png"
image = Image.open(image_path)
# 有的png是4个通道，除了RGB三通道外，还有一个透明度通道，所以调用image =image.convert('RGB')保留其他颜色通道，加上这一步后，可以适应png、jpg各种格式的图片
image = image.convert('RGB')

transf = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                         torchvision.transforms.ToTensor()])

image = transf(image)
model = torch.load("tudui_3.pth")
image = torch.reshape(image, (1, 3, 32, 32))
# 以验证模式开启模型
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))