import cv2
import numpy as np
from matplotlib import pyplot as plt
# 讀取圖片並轉成灰階
image = cv2.imread('./Fig0429(a)(blown_ic).tif')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 傅立葉轉換, 取得 spectrum
ft = np.fft.fft2(image)
ft_shift = np.fft.fftshift(ft)
spectrum = 20*np.log(1+np.abs(ft_shift))

# 建一個 filter H(u,v), 除了中心點設 0, 其餘設 1
mask = np.ones_like(image)
rows, cols = mask.shape
centery = mask.shape[0] // 2
centerx = mask.shape[1] // 2
mask[int(rows/2)][int(cols/2)] = 0

# 將 filter H(u,v) 與 F(u,v) 相乘, 並做反傅立葉轉重建圖片
ft_shift = np.multiply(ft_shift,mask) / 255.0
ift_shift = np.fft.ifftshift(ft_shift)
filtered_image = np.fft.ifft2(ift_shift)

# 以下繪圖
plt.subplot(131),plt.imshow(image, cmap = 'gray')
plt.title('Fig 4.29(a)'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(spectrum, cmap = 'gray')
plt.title('Fig 4.29(b)'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(np.abs(filtered_image), cmap = 'gray')
plt.title('Fig 4.30'), plt.xticks([]), plt.yticks([])
plt.show()