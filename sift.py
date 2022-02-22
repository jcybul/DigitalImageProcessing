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


# Load images
img_night = cv.imread('res/test/00021510/20151102_060125.jpg')
img_day = cv.imread('res/test/00021510/20151108_160137.jpg')

# Match images
[brighter, darker] = match_images(img_night, img_day)

# Concatenate matching results
all_images = np.concatenate(
    (img_night, img_day, brighter, darker), axis=1)


cv.imshow('Matching', all_images)
cv.waitKey(0)
cv.destroyAllWindows()

# # img_night_hsv = cv.cvtColor(img_night, cv.COLOR_RGB2HSV)
# # img_day_hsv = cv.cvtColor(img_day, cv.COLOR_RGB2HSV)

# # night_brightness = img_night_hsv[..., 2].mean()
# # day_brightness = img_day_hsv[..., 2].mean()

# # print(img_night_hsv)
# # print(img_day_hsv)


# #
# # night_brightness = np.average([img_night_hsv[i]
# #                                for i in range(len(img_night_hsv)) if i % 3 == 2])

# # day_brightness = np.average([img_day_hsv[i]
# #                                for i in range(len(img_day_hsv)) if i % 3 == 2])

# print(night_brightness)
# print(day_brightness)


# # cv.imshow('Night', img_night)
# # cv.imshow('Day', img_day)

# img_night_matched = match_histograms(img_day, img_night, channel_axis=-1)

# cv.imshow('Night Matched', img_night_matched)

# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()


# # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# # # (thresh, img_gray) = cv.threshold(
# # #     img_gray, 127, 255, cv.THRESH_BINARY)

# # # SIFT

# # sift = cv.SIFT_create()
# # kp = sift.detect(img_gray, None)

# # img = cv.drawKeypoints(img_gray, kp, img)

# # cv.imwrite("test.jpg", img)

# # Harris
# # img_gray = np.float32(img_gray)
# # dst = cv.cornerHarris(img_gray, 2, 3, 0.04)
# # # result is dilated for marking the corners, not important
# # dst = cv.dilate(dst, None)
# # # Threshold for an optimal value, it may vary depending on the image.
# # img[dst > 0.01*dst.max()] = [0, 0, 255]
# # cv.imshow('dst', img)
# # if cv.waitKey(0) & 0xff == 27:
# #     cv.destroyAllWindows()
