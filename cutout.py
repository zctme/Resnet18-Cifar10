import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Cutout(object):
    """
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img




# if __name__ == '__main__':
#     def tensor2PIL(tensor):  # 将tensor-> PIL
#         unloader = transforms.ToPILImage()
#         image = tensor.cpu().clone()
#         image = unloader(image)
#         return image
#     from PIL import Image
#     import torchvision
#     from  torchvision import transforms
#     path='./imgs/cat1.jpg'
#     imge=Image.open(path)
#     print(type(imge))
#     imge.show()
#     transtensor = torchvision.transforms.ToTensor()
#     img_tensor = transtensor(imge)
#     cutout=Cutout(n_holes=3, length=100)
#     c = cutout(img_tensor)
#     c = tensor2PIL(c)
#     c.show()