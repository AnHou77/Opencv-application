import cv2
import numpy as np
from matplotlib import pyplot as plt

FILTER_RADIUS = 12
GAUSSIAN_KERNEL = (19,19)

# 讀取圖片並轉成灰階
image = cv2.imread('./Fig0431(d)(blown_ic_crop).tif')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# 傅立葉轉換
ft = np.fft.fft2(image)
ft_shift = np.fft.fftshift(ft)

# 定義低通與高通的 filter
mask = np.zeros_like(image)
centerx, centery = mask.shape[0] // 2, mask.shape[1] // 2
cv2.circle(mask, (centerx,centery), FILTER_RADIUS, (255,255,255), -1)[0]
mask2 = 255 - mask
mask = cv2.GaussianBlur(mask, GAUSSIAN_KERNEL, 0)
mask2 = cv2.GaussianBlur(mask2, GAUSSIAN_KERNEL, 0)

# 定義加入 alpha 的 filter (alpha = 0.85, height = 1)
alpha_mask = np.zeros_like(image)
cv2.circle(alpha_mask, (centerx,centery), FILTER_RADIUS, (255*0.15,255*0.15,255*0.15), -1)[0]
alpha_mask = 255 - alpha_mask
alpha_mask = cv2.GaussianBlur(alpha_mask, GAUSSIAN_KERNEL, 0)

# 將 filter 與 傅立葉轉換後的 Fp 相乘
ft_shift_mask1 = np.multiply(ft_shift,mask) / 255.0
ft_shift_mask2 = np.multiply(ft_shift,mask2) / 255.0
ft_shift_mask_alpha = np.multiply(ft_shift,alpha_mask) / 255.0

# 反傅立葉轉換重建圖片
ift_image1 = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_shift_mask1)))
ift_image2 = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_shift_mask2)))
ift_image_alpha = np.abs(np.fft.ifft2(np.fft.ifftshift(ft_shift_mask_alpha)))

# 以下繪圖
plt.subplot(131),plt.imshow(ift_image1, cmap = 'gray')
plt.title('Fig 4.31(d)'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(ift_image2, cmap = 'gray')
plt.title('Fig 4.31(e)'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(ift_image_alpha, cmap = 'gray')
plt.title('Fig 4.31(f)'), plt.xticks([]), plt.yticks([])
plt.show()

