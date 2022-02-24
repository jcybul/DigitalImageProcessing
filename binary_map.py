import numpy as np
import cv2 as cv
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt


def match_images(img1, img2):
    # Convert into HSV
    img1_hsv = cv.cvtColor(img1, cv.COLOR_RGB2HSV)
    img2_hsv = cv.cvtColor(img2, cv.COLOR_RGB2HSV)

    # Calculate brightness
    img1_brightness = img1_hsv[..., 2].mean()
    img2_brightness = img2_hsv[..., 2].mean()

    # Match histograms
    if (img1_brightness > img2_brightness):
        img2_matched = match_histograms(img2, img1, channel_axis=-1)
        return [img1, img2_matched]
    else:
        img1_matched = match_histograms(img1, img2, channel_axis=-1)
        return [img2, img1_matched]


img1 = cv.imread('res/test/00023966/20151120_004607.jpg')
img2 = cv.imread('res/test/00023966/20151119_081621.jpg')
# img1 = cv.imread('res/test/00021510/20151102_060125.jpg')
# img2 = cv.imread('res/test/00021510/20151108_160137.jpg')

img1 = cv.resize(img1, (700, 500), interpolation=cv.INTER_AREA)
img2 = cv.resize(img2, (700, 500), interpolation=cv.INTER_AREA)

img1_night = False
img2_night = False

# Convert into HSV
img1_hsv = cv.cvtColor(img1, cv.COLOR_RGB2HSV)
img2_hsv = cv.cvtColor(img2, cv.COLOR_RGB2HSV)

# Calculate brightness
img1_brightness = img1_hsv[..., 2].mean()
img2_brightness = img2_hsv[..., 2].mean()

scale = 0.2
t11 = img1_brightness * scale
t12 = img1_brightness * scale
t21 = img2_brightness * scale
t22 = img2_brightness * scale

[img1, img2] = match_images(img1, img2)

img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

kernel = np.ones((3, 3), np.float32)/25
img1 = cv.filter2D(img1, -1, kernel)
img2 = cv.filter2D(img2, -1, kernel)

img1 = cv.Canny(image=img1, threshold1=t11, threshold2=t12)
img2 = cv.Canny(image=img2, threshold1=t21, threshold2=t22)
img1 = cv.Canny(image=img1, threshold1=t11, threshold2=t12)
img2 = cv.Canny(image=img2, threshold1=t21, threshold2=t22)

kernel = np.ones((3, 3), np.uint8)
img1 = cv.dilate(img1, kernel, iterations=1)
img2 = cv.dilate(img2, kernel, iterations=1)

img_out = np.concatenate((img1, img2), axis=1)
cv.imshow('Images', img_out)
cv.waitKey(0)

xor = cv.bitwise_xor(img1, img2, mask=None)
xor_not = cv.bitwise_not(xor)

print(cv.countNonZero(xor_not))

cv.imshow('Images', xor_not)
cv.waitKey(0)
