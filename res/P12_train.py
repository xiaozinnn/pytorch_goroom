# 完整的训练套路
import torch
import torchvision
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P12_model import *

# 准备数据集
# 训练集
data_train = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
# 测试集
data_test = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())
# 查看数据集的长度length
data_train_size = len(data_train)
data_test_size = len(data_test)
print("训练数据集的长度为：{}".format(data_train_size))
print("测试数据集的长度为：{}".format(data_test_size))

# 利用dataloader加载数据集
dataloader_train = DataLoader(data_train, batch_size=64)
dataloader_test = DataLoader(data_test, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 创建损失函数-loss
loss_fn = CrossEntropyLoss()

# 定义优化器-optim   SGD：随机梯度下降
# 1e-2:1x(10)^(-2)=0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 记录训练的总轮数
epoch = 10

# 使用tensorboard记录损失率
writer = SummaryWriter("../logs_loss")

# 开始训练
for i in range(epoch):
    print("--------第{}轮训练开始--------".format(i + 1))

    # 训练步骤开始   tranin不是说一定需要写，只对一些特定的有作用，官方文档有写
    tudui.train()
    for data in dataloader_train:
        imgs, targets = data
        output = tudui(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        # 1、清零操作
        optimizer.zero_grad()
        # 2、方向传播
        loss.backward()
        # 3、开始优化
        optimizer.step()

        total_train_step = total_train_step + 1
        # loss加item的话会让输出更干净，如果不加的话Tensor类型会输出例如Tensor(5)的形式，加了会直接输出5
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
            # 增加tensorboard可视化
            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试步骤开始   与tranin一样，eval不是说一定需要写，只对一些特定的有作用，官方文档有写
    tudui.eval()
    # 总体测试集的损失率
    total_test_loss = 0
    # 总体测试集的成功预测数量
    total_accuracy_num = 0
    with torch.no_grad():
        for data in dataloader_test:
            imgs, targets = data
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss.item()

            # 计算一轮预测成功的个数的总和,一定要==targets来进行比较
            accuracy = (output.argmax(1) == targets).sum().item()
            # 相加每一轮的预测成功的个数
            total_accuracy_num = total_accuracy_num + accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy_num/data_test_size))
    total_test_step = total_test_step + 1
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy_num / data_test_size, total_test_step)

    # 保存每轮的模型结果
    torch.save(tudui, "tudui_{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
