from cv2 import StereoBM_PREFILTER_XSOBEL
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


# Load images
img1 = cv.imread('res/test/00023966/20151120_004607.jpg')
img2 = cv.imread('res/test/00023966/20151119_081621.jpg')
# img1 = cv.imread('res/test/00021510/20151102_060125.jpg')
# img2 = cv.imread('res/test/00021510/20151108_160137.jpg')

# Match images
[brighter, darker] = match_images(img1, img2)

matched = np.concatenate((brighter, darker), axis=1)

cv.imshow('Matching', brighter)
cv.waitKey(0)

cv.imshow('Matching', darker)
cv.waitKey(0)
# Denoise images
img1 = cv.fastNlMeansDenoisingColored(brighter, None, 10, 10, 7, 21)
img2 = cv.fastNlMeansDenoisingColored(darker, None, 10, 10, 7, 21)

denoised = np.concatenate((img1, img2), axis=1)
cv.imshow('Matching', denoised)
cv.waitKey(0)


# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0, 0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i] = [1, 0]
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv.DrawMatchesFlags_DEFAULT)
img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
plt.imshow(img3,), plt.show()
