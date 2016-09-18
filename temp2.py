# coding:utf-8

import cv2.cv as cv
# import cv2
import numpy as np

np_gray = np.random.rand(10, 10)
# print(np_gray)


image_cvmat = cv.fromarray(np_gray)
# print(image_cvmat)

# image = cv.CreateImage((100, 100), 8, 1)
image = cv.GetImage(image_cvmat)
print(image)
# image_np = np.asarray(image_color)
# print(image_np)
# cv.CvtColor(image_gray, image_color, cv.CV_GRAY2BGR)

image_final = cv.CreateImage((500, 500), 8, 1)
print(image_final)

cv.Resize(image, image_final)

cv.ShowImage("Focus", image)
cv.WaitKey(0)