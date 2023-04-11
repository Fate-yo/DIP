import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def createBox():
    box = np.zeros((100, 100), np.uint8) + 50
    print(type(box))

    shape = box.shape
    # print(box)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if j in range(0, 100) and i in range(0, 100):
                box[i, j] = 0 if ((i - 25) ** 2 + (
                            j - 25) ** 2 < 25 ** 2 and j < 40) else 190 if j >= 40 and j < 90 and i < 50 else 255
        # if  (i-50)*(i-50)+(j-50)*(j-50)>2500:

    return box


def histogram(image):

    (row, col) = image.shape
    hist = np.zeros(256, np.int32)
    for i in range(row):
        for j in range(col):
            print(image[i,j],end=" ")
            hist[image[i, j]] += 1

    return hist


image0 = createBox()
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(image0, vmin=0, vmax=255, cmap=plt.cm.gray)
plt.title('ideal image')
image_hist0 = histogram(image0)
plt.subplot(1, 2, 2)
plt.bar(range(256), image_hist0,width=2)
plt.show()
