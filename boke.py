import os
import cv2
import numpy as np

# you need this image file
im = cv2.imread("test.jpg", 0)

cv2.imwrite("src.jpg", im)

# x-edge filter
kernelx = np.array([[0, 0, 0],
                    [1, 0, -1],
                    [0, 0, 0]])
edgex = cv2.filter2D(im, -1, kernelx)

# y-edge filter
kernely = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, -1, 0]])
edgey = cv2.filter2D(im, -1, kernely)

cv2.imwrite("edgex.jpg", edgex)
cv2.imwrite("edgey.jpg", edgey)

edgeX = abs(edgex)
edgeY = abs(edgey)

# convert to calcHist supported type
uint16_x = np.uint16(edgeX)
uint16_y = np.uint16(edgeY)

# make histgram
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
for omega in [0, 5, 10, 15, 20, 100]:
    out = 0.1*f*omega + im
    outfile = "out" + str(omega) + ".jpg"
    cv2.imwrite(outfile,out)

