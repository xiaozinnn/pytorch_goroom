# transforms的使用
# transforms提供了一些列的工具，使得数据可以在加载到模型之前被适当地转换和标准化，主要是进行数据预处理和增强。
# 这对于深度学习模型的训练非常重要，因为模型的性能很大程度上依赖与输出的数据
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np

writer = SummaryWriter("logs")

img = Image.open("dataset/train/ants_image/0013035.jpg")

# ToTensor使用：将图片类型转换为ToTensor类型
trans_totensor = transforms.ToTensor()
img_totensor = trans_totensor(img)
writer.add_image("ToTensor", img_totensor)

# Normalize ps:归一化：（原始数据-均值）/标准差
# print(img_totensor[0][0][0])
trans_nor = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_nor = trans_nor(img_totensor)
# print(img_nor[0][0][0])
writer.add_image("normalize", img_nor, 2)

# resize  改变尺寸 ps:先进行resize,后进行ToTensor，分两步进行
print(img.size)  # PIL数据类型，通过resize也是PIL数据类型
trans_resi = transforms.Resize((512, 512))
trans_res_img = trans_resi(img)
# PIL -> Tensor
trans_res_img = trans_totensor(trans_res_img)
writer.add_image("resize", trans_res_img, 0)

# compose 改变尺寸2 ps:利用compose将resize和ToTensor合并在一起执行，只用一步.
#                                      实质是将多个transforms变换组合在一起，并按照顺序执行他们
trans_resi_2 = transforms.Resize(300)
trans_compose = transforms.Compose([trans_resi_2, trans_totensor])
trans_compose_img = trans_compose(img)
print(trans_compose_img.shape)
writer.add_image("Compose", trans_compose_img, 1)

# RandomCrop   随机裁剪，将随机裁剪10次，可以指定W和H，若只指定一个512，则W和H都为512
trans_random = transforms.RandomCrop((300, 400))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
