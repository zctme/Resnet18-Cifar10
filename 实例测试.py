import torch
from PIL import Image
import torchvision
from torch import nn

img_path='./imgs/cat12.jpg'
#读取成PILimge
image=Image.open(img_path)
image=image.convert('RGB')   #有的png格式是四个通道，这样保留其颜色通道

transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                         torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

image=transform(image)
print(image.shape)

#搭建神经网络
model=torchvision.models.resnet18()
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = nn.Linear(512,10)
model.maxpool = nn.Identity()
print(model)
model.load_state_dict(torch.load('models/checkpoint/ckpt_best.pth'))

image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():    #节约内存
    output=model(image)
print(output)
print(output.argmax(1))