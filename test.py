import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 256
transform_test = torchvision.transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
test_data = torchvision.datasets.CIFAR10('./data',train=False,transform=transform_test,download=True)
print(len(test_data))
test_loader = DataLoader(test_data,256)

model = torchvision.models.resnet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Linear(512,10)
model.maxpool = nn.Identity()
print(model)
# 载入权重
model.load_state_dict(torch.load('models/checkpoint/ckpt_best.pth'))
model = model.to(device)

model.eval()
with torch.no_grad():
    total_accuracy = 0
    for img,targets in test_loader:
        img = img.to(device)
        targets = targets.to(device)
        predict = model(img)
        predict = predict.to(device)
        accuracy = (predict.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
    print('准确率:{}%'.format(total_accuracy*100/10000))