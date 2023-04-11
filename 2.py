import cv2
lenna=cv2.imread("1.jpg")
print(type(lenna))
cv2.namedWindow("Lena",cv2.WINDOW_AUTOSIZE)
cv2.imshow("Lena",lenna)

cv2.waitKey(1000)
cv2.destroyWindow("Lena")
cv2.imwrite("test_imwrite.png",lenna,(cv2.IMWRITE_PNG_COMPRESSION,5))

import cv2
import numpy as np

def prewitt(img):
    # 定义Prewitt算子
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    # 使用filter2D函数进行滤波
    dst_x = cv2.filter2D(img, -1, kernel_x)
    dst_y = cv2.filter2D(img, -1, kernel_y)
    # 返回水平和垂直方向上的滤波结果
    return dst_x, dst_y

def sobel(img):
    # 定义Sobel算子
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    # 使用filter2D函数进行滤波
    dst_x = cv2.filter2D(img, -1, kernel_x)
    dst_y = cv2.filter2D(img, -1, kernel_y)
    # 返回水平和垂直方向上的滤波结果
    return dst_x, dst_y

# 读取图像
img = cv2.imread('gg.jpg', 0)
# 对图像进行Prewitt算子滤波
dst_x, dst_y = prewitt(img)
# 对图像进行Sobel算子滤波
#dst_x, dst_y = sobel(img)

# 将水平和垂直方向上的滤波结果合并成一张图像
dst = np.sqrt(np.power(dst_x, 2.0) + np.power(dst_y, 2.0)).astype(np.uint8)

# 显示原图和处理后的图像
cv2.imshow('Original', img)
cv2.imshow('Filtered', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
