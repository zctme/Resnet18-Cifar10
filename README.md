# Resnet18-Cifar10
本文将介绍如何使用数据增强和模型修改的方式，在不使用任何预训练模型的情况下，在ResNet18网络上对Cifar10数据集进行分类任务。
考虑到CIFAR10数据集的图片尺寸太小，ResNet18网络的7x7下采样卷积和池化操作容易丢失一部分信息，所以在实验中我们将7x7的下采样层和最大池化层去掉，替换为一个3x3的下采样卷积，
同时减小该卷积层的步长和填充大小，这样可以尽可能保留原始图像的信息!
在测试集上，我们的模型准确率可以达到96.04%!
