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

# Get pixel position and value


# # Read picture with bad light
# img = cv.imread("res/train/00001323/20151101_221025.jpg")  # Bad Light
# # img = cv.imread("res/train/00011331/20151102_101119.jpg")
# # img = cv.imread("res/train/00011331/20151102_164128.jpg")
# # img = cv.imread("res/train/00011331/20151103_201156.jpg")
# img = cv.imread("res/train/00004232/20151101_072044.jpg")


def get_brightness(img):
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


# img = cv.imread("res/train/00004232/20151101_072044.jpg")
print(get_brightness(img))
