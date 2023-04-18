import os
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import math
from torch.utils.tensorboard import SummaryWriter
from cutout import Cutout
import random
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    np.random.seed(seed)  # np随机数组种子
    random.seed(seed)  # 随机数种子

    torch.backends.cudnn.benchmark = False  # 是否自动加速，自动选择合适算法，false选择固定算法
    torch.backends.cudnn.deterministic = True  # 为了消除该算法本身的不确定性
def rule(epoch):
    if epoch < 5:
        lambd=1
    elif 5<= epoch < 10:
        lambd=5
    elif 10 <= epoch < 15:
        lambd=10
    elif 15 <= epoch < 20:
        lambd = 1
    elif 20 <= epoch < 25:
        lambd = 0.1
    else:
        lambd=0.01

    return lambd
# 设置随机数种子
setup_seed(20)
# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据增强
transform_train = torchvision.transforms.Compose([transforms.RandomCrop(32,padding=4),#将尺寸为32x32的图像填充为40x40，然后随机裁剪成32x32
                                                  transforms.RandomHorizontalFlip(), #随机翻转
                                                  transforms.ToTensor(),#（H，W，C)-->（C，H，W）
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                  Cutout(n_holes=1, length=16)])
transforms_test = torchvision.transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])#从数据集中随机抽样计算得到的。

# 读数据
train_data = torchvision.datasets.CIFAR10('./data',train=True,transform=transform_train,download=True)
test_data = torchvision.datasets.CIFAR10('./data',train=False,transform=transforms_test,download=True)
len_train = len(train_data)
len_test = len(test_data)

# 利用Dataloader来加载数据
train_load = DataLoader(train_data,batch_size=256,shuffle=True)
test_load = DataLoader(test_data,batch_size=256)


# 搭建神经网络
model = torchvision.models.resnet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Linear(512,10)
model.maxpool = nn.Identity()#删除最大池化层
model = model.to(device)

# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

# 设置训练网络的一些参数
# 学习率
lr = 0.1
# 回合数
epoch = 35
# 存放准确率的列表
Accuracy = [0]

optimer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimer, lr_lambda=rule,last_epoch=-1)
print("初始化的学习率：", optimer.defaults['lr'])
# 添加tensorboard
writer = SummaryWriter("./logs_train")
for i in range(epoch):
    train_loss = 0
    test_loss = 0

    model.train() # 训练步骤开始
    for imgs,targets in train_load:
        imgs = imgs.to(device)
        targets = targets.to(device)
        predict = model(imgs)
        predict = predict.to(device)
        loss = criterion(predict,targets)
        train_loss += loss
        # 优化器优化模型
        optimer.zero_grad() # 清空过往梯度
        loss.backward() # 反向传播，计算当前每个参数的梯度梯度
        optimer.step() # 根据梯度更新网络参数
    print('第{}轮训练损失为{}'.format(i+1,train_loss/len(train_load)))
    print("第%d个epoch的学习率：%f" % (i+1, optimer.param_groups[0]['lr']))
    scheduler.step()   # 放在for循环外,更新lr
    writer.add_scalar("训练损失", train_loss/len(train_load), i)

    model.eval() # 测试步骤开始
    with torch.no_grad():# 我们只是想看一下训练的效果，并不是想通过测试集来更新网络
        total_accuracy = 0
        for imgs,targets in test_load:
            imgs = imgs.to(device)
            targets = targets.to(device)
            predict = model(imgs)
            loss = criterion(predict,targets)
            test_loss += loss
            accuracy = (predict.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy
        print('第{}轮验证损失为:{}，准确率:{}%'.format(i + 1, test_loss/len(test_load),total_accuracy*100/len_test))
        writer.add_scalar("准确率", total_accuracy*100/len_test, i)
        writer.add_scalar("测试损失", test_loss/len(test_load), i)


    if not os.path.isdir("./models/checkpoint"):
        os.mkdir("./models/checkpoint")
    if total_accuracy*100/len_test>max(Accuracy):
        torch.save(model.state_dict(),'./models/checkpoint/ckpt_best.pth')
        print('-------最优模型已保存-------')
        Accuracy.append(total_accuracy*100/len_test)
        print(Accuracy)
        print(max(Accuracy))
    torch.save(model.state_dict(), './models/checkpoint/ckpt_last.pth')
writer.close()













































































































