import cv2
import numpy as np
from matplotlib import pyplot as plt
# 讀取圖片並轉成灰階
image = cv2.imread('./Fig0427(a)(woman).tif')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image_rect = cv2.imread('./Fig0424(a)(rectangle).tif')
image_rect = cv2.cvtColor(image_rect,cv2.COLOR_BGR2GRAY)
image_rect = cv2.resize(image_rect,(512,512))

# 將 woman 的圖片做傅立葉轉換, 並取得 phase angle
ft_woman = np.fft.fft2(image)
ft_shift_woman = np.fft.fftshift(ft_woman)
origin_angle_woman = np.angle(ft_shift_woman)
angle_woman = np.exp(origin_angle_woman * 1j)

# 用 phase angle 做反傅立葉轉換重建圖片
reconstructed = np.fft.ifftshift(angle_woman)
reconstructed = np.fft.ifft2(reconstructed)
# 取得 woman image 傅立葉轉換後的 spectrum
spectrum = 20*np.log(1+np.abs(ft_shift_woman))

# 同樣將 rectangle image 做傅立葉轉換及取得 phase angle
ft_rect = np.fft.fft2(image_rect)
ft_shift_rect = np.fft.fftshift(ft_rect)
angle_rect = np.angle(ft_shift_rect)
angle_rect = np.exp(angle_rect * 1j)

# 利用 woman's phase angle & rectangle's spectrum 重建圖e
# 利用 rectangle's phase angle & woman's spectrum 重建圖f
combined_e = angle_woman * np.abs(ft_shift_rect) / 255.0
reconstructed_e = np.abs(np.fft.ifft2(np.fft.ifftshift(combined_e)))
combined_f = angle_rect * np.abs(ft_shift_woman) / 255.0
reconstructed_f = np.abs(np.fft.ifft2(np.fft.ifftshift(combined_f)))

# 以下繪圖
plt.subplot(231),plt.imshow(image, cmap = 'gray')
plt.title('Fig 4.27(a)'), plt.xticks([]), plt.yticks([])
plt.subplot(232),plt.imshow(np.abs(origin_angle_woman), cmap = 'gray')
plt.title('Fig 4.27(b)'), plt.xticks([]), plt.yticks([])
plt.subplot(233),plt.imshow(np.abs(reconstructed), cmap = 'gray')
plt.title('Fig 4.27(c)'), plt.xticks([]), plt.yticks([])
plt.subplot(234),plt.imshow(spectrum, cmap = 'gray')
plt.title('Fig 4.27(d)'), plt.xticks([]), plt.yticks([])
plt.subplot(235),plt.imshow(reconstructed_e, cmap = 'gray')
plt.title('Fig 4.27(e)'), plt.xticks([]), plt.yticks([])
plt.subplot(236),plt.imshow(reconstructed_f, cmap = 'gray')
plt.title('Fig 4.27(f)'), plt.xticks([]), plt.yticks([])
plt.show()