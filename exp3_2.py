import numpy as np
import cv2
from matplotlib import pyplot as plt

# %matplotlib inline
# %config InlinBackend. figure format-"retina"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负
# openCV函数实现傅里叶变换
img = cv2.imread(r'1.jpg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
# dft1=cv2. shift_dft(dft)
# dft[:,:,0]为傅里叶变换的实部，dft[:,:,1]为傅里叶变换的虚部
magnitude0 = cv2.magnitude(dft[:, :, 0], dft[:, :, 1])  # 幅值谱
magnitude1 = 20 * np.log(1 + magnitude0)  # 腐值谱
phase_angle0 = cv2.phase(dft[:, :, 0], dft[:, :, 1])  # 相位谱
img_back = cv2.idft(dft)
img_back1 = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
plt.figure(figsize=(10, 6))
plt.subplot(151), plt.imshow(img, cmap='gray')
plt.title("原图像"), plt.axis('off')
plt.subplot(152), plt.imshow(magnitude0, cmap='gray')
plt.title("幅值谱"), plt.axis('off')
plt.subplot(153), plt.imshow(magnitude1, cmap='gray')
plt.title("对数变换后的幅值谱"), plt.axis('off')
plt.subplot(154), plt.imshow(phase_angle0, cmap='gray')
plt.title("相位谱"), plt.axis('off')
plt.subplot(155), plt.imshow(img_back1, cmap='gray')
plt.title("重构图像"), plt.axis('off')
plt.show()
