# transforms的介绍----
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

path = 'dataset/train/ants_image/0013035.jpg'
img_path = Image.open(path)
# print(img_path)

writer = SummaryWriter("logs")

# 1、transforms如何被使用
tensor_tran = transforms.ToTensor()
tersor_img = tensor_tran(img_path)

writer.add_image("terson_img",tersor_img)

writer.close()