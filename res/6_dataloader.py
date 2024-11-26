# dataloader的使用
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10('./datasets', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# DataLoader 参数1：数据集; 参数2 batch_size：每次拉多少数据；参数3 shuffle:每次是否是随机拉取；参数4 num_workers：是否是主线程运行；参数5：drop_last:是否保留取余的余数
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试集中第一张图片的尺寸以及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("loder")
for i in range(2):
    temp = 0
    for data in test_loader:
        (imgs, targets) = data
        writer.add_images("loader_{}".format(i), imgs, temp)
        temp = temp + 1

writer.close()
