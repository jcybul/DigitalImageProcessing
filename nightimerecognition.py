from time import sleep
from matplotlib import pyplot as plt
from cv2 import StereoBM_PREFILTER_XSOBEL
import numpy as np
import cv2 as cv
import skimage

def edgeDetection(i1, i2):
    img1 = cv.imread(i1, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(i2, cv.IMREAD_GRAYSCALE)

    wide = cv.Canny(img1, 10, 200)
    mid = cv.Canny(img2, 30, 150)
    plt.subplot(2, 2, 1)
    plt.imshow(wide)

    plt.subplot(2, 2, 2)
    plt.imshow(mid)
    plt.title("Edge Detection")
    plt.show()


def newEdgeDection(img1,img2):
    img1_night = False
    img2_night = False

    #MAtch images
    l = match_images(img1, img2)
    img1 = l[0]
    img2 = l[2]
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

    return img1,img2



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


def extractFeatures(i1, i2):
    img1 = cv.imread(i1, cv.IMREAD_GRAYSCALE)  # queryImage
    img2 = cv.imread(i2, cv.IMREAD_GRAYSCALE)

    img1,img2 = newEdgeDection(img1,img2)


    # (thresh, img1) = cv.threshold(img1, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # (thresh, img2) = cv.threshold(img2, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # trainImage# Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good = 0
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            good = good + 1

    print("Ratioed Matches", good)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    print("Descriptors 2:  ", len(des2))
    print("Descriptors 1:  ", len(des1))
    print("Key points 1:", len(kp1))
    print("Key points 2:", len(kp2))

    # plt.imshow(img3), plt.show()
    total = len(des1) + len(des2)

    return total, good, img1, img2, img3


if __name__ == '__main__':

    # edgeDetection("res/p1.jpg", "res/p2.jpg")
    # sleep(10000)
    total, num, img1, img2, img3 = extractFeatures("res/p1.jpg", "res/p3.jpg")
    if (total / 2) * 0.4 < num:
        plt.subplot(2, 2, 1)
        plt.imshow(img1)

        plt.subplot(2, 2, 2)
        plt.imshow(img2)
        plt.subplot(2, 2, 3)
        plt.imshow(img3)
        plt.title("Same Place")
        plt.show()
    else:

        plt.subplot(2, 2, 1)
        plt.imshow(img1)

        plt.subplot(2, 2, 2)
        plt.imshow(img2)
        plt.title("Different Place")
        plt.show()

    # print(extract_features("res/p1.jpg"))
    exit(0)
