import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import cv2 as cv
import matplotlib.pyplot as plt

def gradient_one(img):
    # img = img.squeeze(0)
    ten=torch.unbind(img)
    x=ten[0].unsqueeze(0).unsqueeze(0)

    a=0.25 * np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    G_x=conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    b= 0.25 * np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    G_y=conv2(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G

def gradient_rgb(img):
    res = torch.stack([gradient_one(img[i,:,:].unsqueeze(0)) for i in range(3)]).squeeze()
    return res

def BasicImageDataset(path, img_size):
    return datasets.ImageFolder(root = path,
                                transform = transforms.Compose([
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float),
                                    ]))

if __name__ == '__main__':
    data = BasicImageDataset("res/test", (200,300))
    img = data[0][0]
    img = gradient_rgb(img).numpy().transpose(1,2,0)
    plt.imshow(img)
    plt.show()
