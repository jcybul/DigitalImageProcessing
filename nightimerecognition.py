from time import sleep
from matplotlib import pyplot as plt
from cv2 import StereoBM_PREFILTER_XSOBEL
import numpy as np
import cv2 as cv
import skimage
from skimage.exposure import match_histograms, exposure


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


def newEdgeDection(img1, img2):
    img1_night = False
    img2_night = False

    # MAtch images

    l = match_images(img1, img2)
    img1 = l[0]
    img2 = l[1]

    t11 = 200
    t12 = 150
    t21 = 200
    t22 = 150

    if img1_night:
        t11 = 200
        t12 = 100
    if img2_night:
        t21 = 140
        t22 = 150

    img1 = cv.Canny(image=img1, threshold1=t11, threshold2=t12)
    img2 = cv.Canny(image=img2, threshold1=t21, threshold2=t22)

    return img1, img2



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


def creakDownImage(img,n):
    img = cv.resize(img, (300, 200), interpolation=cv.INTER_AREA)
    h, w, c = img.shape

    wx = int(h/n)
    wy = int(w/n)
    # print(wx)
    ret = []
    i = 0
    j = 0
    for _ in range(n):
        for _ in range(n):
            ret.append(img[i:i+wx, j:j+wy])
            # print("x end :",i+wx)
            # print("y end:",j+wy)
            j = j+wy
        i = i+wx
        j = 0
    return ret


def extractFeatures(img1, img2,n):
    #img1 = cv.imread(i1)  # queryImage
    #img2 = cv.imread(i2)

    #img1,img2 = newEdgeDection(img1,img2)
    for i in range(n):
        l = match_images(img1, img2)
        img1 = l[0]
        img2 = l[1]
    # img1 = cv.imwrite('res/test1.jpg',img1)
    # img2 = cv.imwrite("res/test2.jpg",img2)
    # img1 = cv.imread("res/test1.jpg", 0)
    # img2 = cv.imread("res/test2.jpg", 0)


    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 5
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=5)  # or pass empty dictionary

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

    print("Good Matches", good)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    # print("Descriptors 2:  ", len(des2))
    # print("Descriptors 1:  ", len(des1))
    # print("Key points 1:", len(kp1))
    # print("Key points 2:", len(kp2))

    # plt.imshow(img3), plt.show()
    total = len(des1) + len(des2)

    return total, good, img1, img2, img3

if __name__ == '__main__':
    count = 0
    totalF = 0
    n = 2
    one = cv.imread("res/p2.jpg")
    two = cv.imread("res/p3.jpg")
    img1 = creakDownImage(one,n)
    img2 = creakDownImage(two,n)

    total, num, im, im2, img = extractFeatures(one,two,10)
    for i in range(n*n):
        total, temp_num, temp, temp2, img3 = extractFeatures(img1[i],img2[i],10)
        totalF = temp_num + totalF
        plt.subplot(2, 2, 1)
        plt.imshow(temp)
        plt.subplot(2, 2, 2)
        plt.imshow(temp2)
        plt.subplot(2, 2, 3)
        plt.imshow(img3)
        plt.show()


    print("Non break total = ", num)
    print("Separate matching = ", totalF)
    plt.subplot(2, 2, 1)
    plt.imshow(im)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(img)
    plt.show()

    # edgeDetection("res/p1.jpg", "res/p2.jpg")
    # sleep(10000)

    # print(extract_features("res/p1.jpg"))
    exit(0)
