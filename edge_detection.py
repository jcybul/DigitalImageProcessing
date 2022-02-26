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
import PIL


def get_brightness(img):

    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    w, h, c = img.shape

    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.GaussianBlur(mask, (11, 11), 0)
    mask = cv.threshold(mask, 200, 255, cv.THRESH_BINARY)[1]
    bright_num = cv.countNonZero(mask)
    mask = cv.bitwise_not(mask)
    img = cv.bitwise_and(img, img, mask=mask)
    bright_sum = cv.cvtColor(img, cv.COLOR_BGR2HSV)[..., 2].sum()
    pixel_num = w * h
    brightness = bright_sum/(pixel_num-bright_num)

    return brightness


def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def gradient_one(img, brightness):
    ten = torch.unbind(img)

    x = ten[0].unsqueeze(0).unsqueeze(0)
    # Want lower number with higher bightness
    scale = 10

    a = scale/brightness * np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(
        a).float().unsqueeze(0).unsqueeze(0))
    G_x = conv1(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    b = scale/brightness * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(
        b).float().unsqueeze(0).unsqueeze(0))
    G_y = conv2(Variable(x)).data.view(1, x.shape[2], x.shape[3])

    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G


def gradient_rgb(img):
    brightness = get_brightness(img)
    img = apply_transform(img, (400, 600))
    res = torch.stack([gradient_one(img[i, :, :].unsqueeze(0), brightness)
                       for i in range(3)]).squeeze()
    return res


def apply_transform(img, img_size):
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.ConvertImageDtype(torch.float),
                                    ])
    img = transform(img)
    return img


def BasicImageDataset(path):
    return datasets.ImageFolder(root=path)


def CVImage(img):
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)


def process_image(img):
    img = gradient_rgb(img).numpy().transpose(1, 2, 0)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    (thresh, img) = cv.threshold(img, 0.1, 1, cv.THRESH_BINARY)
    kernel = np.ones((4, 4), np.uint8)
    img = cv.dilate(img, kernel, iterations=1)
    return img

# Both 0 and Both 1

# 1 0 and 0 1
# 1 0 and 1 0
# A and B
# not A and not B


def match_images(img1, img2):
    ones = cv.countNonZero(cv.bitwise_and(img1, img2))
    img1n = cv.bitwise_not(img1)
    img2n = cv.bitwise_not(img2)
    zeros = cv.countNonZero(cv.bitwise_and(img1n, img2n))
    return ones+zeros


if __name__ == '__main__':
    data = BasicImageDataset("res/test")
    master = data[17][0]
    master = process_image(master)
    results = []
    for i in range(len(data)):
        img = data[i][0]
        img = process_image(img)
        result = match_images(master, img)
        results.append((result, i))
    results = sorted(results, key=lambda tup: tup[0])
    print(results)
