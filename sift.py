from cv2 import StereoBM_PREFILTER_XSOBEL
import numpy as np
import cv2 as cv
from skimage.exposure import match_histograms


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


def sift_feature_extraction(img1, img2, brighter, darker):
    # Create SIFT extractor
    sift = cv.SIFT_create()

    # Extract keypoints
    kp1 = sift.detect(img1, None)
    kp2 = sift.detect(img2, None)
    kpb = sift.detect(brighter, None)
    kpd = sift.detect(darker, None)

    # Draw keypoints on image
    img1_kp = cv.drawKeypoints(img1, kp1, img1)
    img2_kp = cv.drawKeypoints(img2, kp2, img2)
    brighter_kp = cv.drawKeypoints(brighter, kpb, brighter)
    darker_kp = cv.drawKeypoints(darker, kpd, darker)

    return [img1_kp, img2_kp, brighter_kp, darker_kp]


def sobel_edge_detection(img1, img2, brighter, darker):
    sobel1 = cv.Sobel(src=img1, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    sobel2 = cv.Sobel(src=img2, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    sobelb = cv.Sobel(src=brighter, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    sobeld = cv.Sobel(src=darker, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    return [sobel1, sobel2, sobelb, sobeld]


def canny_edge_detection(img1, img2, brighter, darker):
    canny1 = cv.Canny(image=img1, threshold1=100, threshold2=200)
    canny2 = cv.Canny(image=img2, threshold1=100, threshold2=200)
    cannyb = cv.Canny(image=brighter, threshold1=100, threshold2=200)
    cannyd = cv.Canny(image=darker, threshold1=100, threshold2=200)
    return [canny1, canny2, cannyb, cannyd]


# Load images
img1 = cv.imread('res/test/00021510/20151102_060125.jpg')
img2 = cv.imread('res/test/00021510/20151108_160137.jpg')

# Match images
[brighter, darker] = match_images(img1, img2)

# Concatenate matching results
all_images = np.concatenate(
    (img1, img2, brighter, darker), axis=1)

cv.imshow('Matching', all_images)
cv.waitKey(0)

# Appraoch #0.4: Erosion
# kernel = np.ones((4, 4), np.uint8)
# img1 = cv.erode(img1, kernel)
# img2 = cv.erode(img2, kernel)
# brighter = cv.erode(brighter, kernel)
# darker = cv.erode(darker, kernel)

# Approach #0.5: Convert into grayscale
# img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# brighter = cv.cvtColor(brighter, cv.COLOR_BGR2GRAY)
# darker = cv.cvtColor(darker, cv.COLOR_BGR2GRAY)

# Approach #1: SIFT Feature extraction
[img1_kp, img2_kp, brighter_kp, darker_kp] = sift_feature_extraction(
    img1, img2, brighter, darker)

all_kp = np.concatenate(
    (img1_kp, img2_kp, brighter_kp, darker_kp), axis=1)

cv.imshow('Matching', all_kp)
cv.waitKey(0)

# Approach #1.5: Blur the image
# brighter = cv.GaussianBlur(brighter, (3, 3), 0, cv.BORDER_DEFAULT)
# darker = cv.GaussianBlur(darker, (3, 3), 0, cv.BORDER_DEFAULT)

# Approach #2: Sobel Edge Detection
[sobel1, sobel2, sobelb, sobeld] = sobel_edge_detection(
    img1, img2, brighter, darker)

all_sobel = np.concatenate(
    (sobel1, sobel2, sobelb, sobeld), axis=1)

# cv.imshow('Matching', all_sobel)
# cv.waitKey(0)

# Approach 2.5.1: Downsample pyrDown
# img1 = cv.pyrDown(img1)
# img2 = cv.pyrDown(img2)
# brighter = cv.pyrDown(brighter)
# darker = cv.pyrDown(darker)

# Approach 2.5.2: Downsample resize
# ratio = 0.5
# img1 = cv.resize(img1, (0, 0), fx=ratio, fy=ratio,
#                  interpolation=cv.INTER_NEAREST)
# img2 = cv.resize(img2, (0, 0), fx=ratio, fy=ratio,
#                  interpolation=cv.INTER_NEAREST)
# brighter = cv.resize(brighter, (0, 0), fx=ratio, fy=ratio,
#                      interpolation=cv.INTER_NEAREST)
# darker = cv.resize(darker, (0, 0), fx=ratio, fy=ratio,
#                    interpolation=cv.INTER_NEAREST)

# Approach #3: Canny Edge Detection
[canny1, canny2, canny1b, cannyd] = canny_edge_detection(
    img1, img2, brighter, darker)

all_canny = np.concatenate(
    (canny1, canny2, canny1b, cannyd), axis=1)

cv.imshow('Matching', all_canny)
cv.waitKey(0)

# Approach #1: SIFT Feature extraction
[img1_kp, img2_kp, brighter_kp, darker_kp] = sift_feature_extraction(
    canny1, canny2, canny1b, cannyd)

all_kp = np.concatenate(
    (img1_kp, img2_kp, brighter_kp, darker_kp), axis=1)

cv.imshow('Matching', all_kp)
cv.waitKey(0)
cv.destroyAllWindows()
