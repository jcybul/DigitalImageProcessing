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

img1_night = False
img2_night = False

# Convert into HSV
img1_hsv = cv.cvtColor(img1, cv.COLOR_RGB2HSV)
img2_hsv = cv.cvtColor(img2, cv.COLOR_RGB2HSV)

# Calculate brightness
img1_brightness = img1_hsv[..., 2].mean()
img2_brightness = img2_hsv[..., 2].mean()

if img1_brightness < 50:
    img1_night = True
if img2_brightness < 50:
    img2_night = True

img1 = cv.resize(img1, (300, 200), interpolation=cv.INTER_AREA)
img2 = cv.resize(img2, (300, 200), interpolation=cv.INTER_AREA)

t11 = 100
t12 = 150
t21 = 100
t22 = 150

if img1_night:
    t11 = 40
    t12 = 50
if img2_night:
    t21 = 40
    t22 = 50


img1 = cv.Canny(image=img1, threshold1=t11, threshold2=t12)
img2 = cv.Canny(image=img2, threshold1=t21, threshold2=t22)

img_out = np.concatenate((img1, img2), axis=1)
cv.imshow('Images', img_out)
cv.waitKey(0)


# b1 = img1.copy()
# b1[:, :, 1] = 0
# b1[:, :, 2] = 0

# g1 = img1.copy()
# g1[:, :, 0] = 0
# g1[:, :, 2] = 0

# r1 = img1.copy()
# r1[:, :, 0] = 0
# r1[:, :, 1] = 0

# # Second Image
# b2 = img2.copy()
# b2[:, :, 1] = 0
# b2[:, :, 2] = 0

# g2 = img2.copy()
# g2[:, :, 0] = 0
# g2[:, :, 2] = 0

# r2 = img2.copy()
# r2[:, :, 0] = 0
# r2[:, :, 1] = 0

# t11 = 100
# t12 = 150
# t21 = 100
# t22 = 150

# if img1_night:
#     t11 = 40
#     t12 = 80
# if img2_night:
#     t21 = 40
#     t22 = 80

# b1 = cv.Canny(image=b1, threshold1=t11, threshold2=t12)
# g1 = cv.Canny(image=g1, threshold1=t11, threshold2=t12)
# r1 = cv.Canny(image=r1, threshold1=t11, threshold2=t12)

# b2 = cv.Canny(image=b2, threshold1=t21, threshold2=t22)
# g2 = cv.Canny(image=g2, threshold1=t21, threshold2=t22)
# r2 = cv.Canny(image=r2, threshold1=t21, threshold2=t22)

# img_out1 = np.concatenate((b1, g1, r1), axis=1)
# img_out2 = np.concatenate((b2, g2, r2), axis=1)

# img_out = np.concatenate((img_out1, img_out2), axis=1)
# cv.imshow('Images', img_out)
# cv.waitKey(0)
