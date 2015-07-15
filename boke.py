import os
import cv2
import numpy as np

# you need this image file
im_ori = cv2.imread("input.jpg", 0)
im = np.int16(im_ori)

cv2.imwrite("src.jpg", im)

# x-edge filter
kernelx = np.array([[1, 0, -1]])

# y-edge filter
kernely = np.array([[1],
                    [0],
                    [-1]])

edgex = cv2.filter2D(im, -1, kernelx)
edgey = cv2.filter2D(im, -1, kernely)

absx = abs(edgex)
absy = abs(edgey)

cv2.imwrite("edgex.jpg", absx)
cv2.imwrite("edgey.jpg", absy)

# convert to calcHist supported type
uint16_x = np.uint16(absx)
uint16_y = np.uint16(absy)

# make edge strength histgram
hist = cv2.calcHist([uint16_x, uint16_y],[0],None,[512],[0,512])

f = open("hist.txt", "w")
for i,x in enumerate(hist):
    s = '%d %d\n' % (i, x)
    f.write(s)
f.close()

# sharpen filter
fil = np.array([[-2, 1, -2],
                [1,  4,  1],
                [-2, 1, -2]])
f = cv2.filter2D(im, -1, fil)

# several weight of filterd values
for weight in [0, 5, 10, 15, 20, 100]:
    out = 0.1*f*weight + im
    outfile = "out" + str(weight) + ".jpg"
    cv2.imwrite(outfile,out)

