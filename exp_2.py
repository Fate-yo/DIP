import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']


def link(r, a, b, c, d):
    ab = [a, b]
    cd = [c, d]
    s = ((cd[1] - cd[0]) / (ab[1] - ab[0])) * (r - ab[0]) + cd[0]
    return s


def global_linear_transmation(im, c=0, d=255):
    img = im.copy()
    maxV = img.max()
    minV = img.min()
    if maxV == minV:
        return np.uint8(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((d - c) / (maxV - minV)) * (img[i, j] - minV) + c
            return np.uint8(img)


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


img = cv2.imread('1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(4, 3, 1)
dark_img = gamma_trans(img, 4)
plt.title("变暗后")
plt.imshow(dark_img)

plt.subplot(4, 3, 2)
light_img = gamma_trans(img, 0.4)
plt.title("变亮后")
plt.imshow(light_img)

plt.subplot(4, 3, 3)
low_contrast_img = cv2.convertScaleAbs(img, alpha=0.5, beta=128)
plt.title("降低对比度")
plt.imshow(low_contrast_img)

plt.subplot(4, 3, 4)
hist1 = cv2.calcHist([dark_img], [0], None, [256], [0, 255])
plt.plot(hist1)

plt.subplot(4, 3, 5)
hist2 = cv2.calcHist([light_img], [0], None, [256], [0, 255])
plt.plot(hist2)

plt.subplot(4, 3, 6)
hist3 = cv2.calcHist([low_contrast_img], [0], None, [256], [0, 255])
plt.plot(hist3)

plt.subplot(4, 3, 7)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray_img)
plt.title("均衡化处理")
plt.imshow(equalized_img)

plt.subplot(4, 3, 8)
plt.title("全局线性变换")
globals_transform = global_linear_transmation(img, d=222)
plt.imshow(globals_transform)

plt.subplot(4, 3, 9)
gamma_ = gamma_trans(img, 1)
plt.title("gamma")
plt.imshow(gamma_)
plt.subplot(4, 3, 11)
hist4 = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
plt.plot(hist4)
plt.subplot(4, 3, 12)
hist5 = cv2.calcHist([globals_transform], [0], None, [256], [0, 255])
plt.plot(hist5)
plt.subplot(4, 3, 10)
hist6 = cv2.calcHist([gamma_], [0], None, [256], [0, 255])
plt.plot(hist6)
plt.show()


def addSaltAndPepper(src, percentage):
    NoiseImg = src.copy()
    NoiseNum = int(percentage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        if random.randint(0, 1) == 0:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


def addGaussianNoise(src, means, sigma):
    NoiseImg = src / src.max()
    rows = NoiseImg.shape[0]
    cols = NoiseImg.shape[1]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[i, j] = NoiseImg[i, j] + random.gauss(means, sigma)
            if (NoiseImg[i, j] < 0).all():
                NoiseImg[i, j] = 0
            elif (NoiseImg[i, j] > 1).all():
                NoiseImg[i, j] = 1
    NoiseImg = np.uint8(NoiseImg * 255)
    return NoiseImg


SaltAndPepper = addSaltAndPepper(img, 0.05)
GaussianNoise = addGaussianNoise(img, 0, 0.05)


def prewitt(img):
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    img_prewitt_x = cv2.filter2D(img, -1, kernel_x)
    img_prewitt_y = cv2.filter2D(img, -1, kernel_y)
    img_prewitt = cv2.addWeighted(img_prewitt_x, 0.5, img_prewitt_y, 0.5, 0)

    return img_prewitt


def sobel(img):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]])
    img_sobel_x = cv2.filter2D(img, -1, kernel_x)
    img_sobel_y = cv2.filter2D(img, -1, kernel_y)
    img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)
    return img_sobel


plt.subplot(1, 2, 1)
plt.title("椒盐噪声")
plt.imshow(SaltAndPepper)
plt.subplot(1, 2, 2)
plt.title("高斯噪声")
plt.imshow(GaussianNoise)

plt.show()
plt.subplot(2, 6, 1)
plt.title("椒盐噪声")
plt.imshow(SaltAndPepper)
plt.subplot(2, 6, 7)
plt.imshow(GaussianNoise)

blur_1 = cv2.blur(SaltAndPepper, (3, 3))

plt.subplot(2, 6, 2)
plt.title("均值滤波")
plt.imshow(blur_1)
medianBlur_1 = cv2.medianBlur(SaltAndPepper, 3)
plt.subplot(2, 6, 3)
plt.title("中值滤波")
plt.imshow(medianBlur_1)
SaltAndPepper_1 = cv2.GaussianBlur(SaltAndPepper, (3, 3), 1)
plt.subplot(2, 6, 4)
plt.title("高斯滤波")
plt.imshow(SaltAndPepper_1)
prewitt_1 = prewitt(SaltAndPepper)
plt.subplot(2, 6, 5)
plt.title("sobel")
plt.imshow(prewitt_1)
sobel_2 = sobel(SaltAndPepper)
plt.subplot(2, 6, 6)
plt.title("sobel")
plt.imshow(sobel_2)

blur_2 = cv2.blur(GaussianNoise, (3, 3))
plt.subplot(2, 6, 8)
plt.title("均值滤波")
plt.imshow(blur_2)
medianBlur_2 = cv2.medianBlur(GaussianNoise, 3)
plt.subplot(2, 6, 9)
plt.title("中值滤波")
plt.imshow(medianBlur_2)
SaltAndPepper_2 = cv2.GaussianBlur(GaussianNoise, (3, 3), 1)
plt.subplot(2, 6, 10)
plt.title("高斯滤波")
plt.imshow(SaltAndPepper_2)

prewitt_2 = prewitt(GaussianNoise)
plt.subplot(2, 6, 11)
plt.title("prewitt")
plt.imshow(prewitt_2)
sobel_2 = sobel(GaussianNoise)
plt.subplot(2, 6, 12)
plt.title("sobel")
plt.imshow(sobel_2)
plt.show()
