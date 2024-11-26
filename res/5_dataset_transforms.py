# 数据集与transforms的连用
import torchvision
from torch.utils.tensorboard import SummaryWriter

datasets_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./datasets", train=True, transform=datasets_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=datasets_transforms, download=True)
# # 打印所有类别  ps:相当于多分类
# print(test_set.classes)
# # 将测试集的第一张图片取出来
# img, target = test_set[0]
# # 打印test_set[0]这个数据属于哪一个类别
# print(test_set.classes[target])
# # 展示测试集的第一张图片
# img.show()

writer = SummaryWriter("p5")
for i in range(10):
    img,target = test_set[i]
    writer.add_image("test_set",img,i)

writer.close()