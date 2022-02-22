import torch.nn.functional as F
from torch import Tensor
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms


def normalize(img):
    return F.normalize(img, p=1, dim=0)

def gradient_one(img):
    # img = img.squeeze(0)
    ten=torch.unbind(img)
    x=ten[0].unsqueeze(0).unsqueeze(0)

    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    G_x=conv1(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    G_y=conv2(Variable(x)).data.view(1,x.shape[2],x.shape[3])

    G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G

def gradient_rgb(img):
    res = torch.stack([gradient_one(img[i,:,:].unsqueeze(0)) for i in range(3)]).squeeze()
    return res
