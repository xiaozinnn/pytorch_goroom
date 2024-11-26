# tensorboard可视化的使用
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "dataset/train/ants_image/116570827_e9c126745d.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("tran", img_array, 3, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x", 2 * i, i)

writer.close()
