import numpy as np
import cv2 as cv

img = cv.imread('res/test/00021510/20151102_060125.jpg')
# img = cv.imread('res/test/00023966/20151118_231631.jpg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
(thresh, img_gray) = cv.threshold(
    img_gray, 127, 255, cv.THRESH_BINARY)

# SIFT

sift = cv.SIFT_create()
kp = sift.detect(img_gray, None)

img = cv.drawKeypoints(img_gray, kp, img)

cv.imwrite("test.jpg", img)

# Harris
# img_gray = np.float32(img_gray)
# dst = cv.cornerHarris(img_gray, 2, 3, 0.04)
# # result is dilated for marking the corners, not important
# dst = cv.dilate(dst, None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst > 0.01*dst.max()] = [0, 0, 255]
# cv.imshow('dst', img)
# if cv.waitKey(0) & 0xff == 27:
#     cv.destroyAllWindows()
