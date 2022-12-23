import cv2
import numpy as np
from matplotlib import pyplot as plt
# 讀取 M x N 圖片(image(a)) 並轉成灰階
image = cv2.imread('./Fig0431(d)(blown_ic_crop).tif')
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# padding image(a) to image(b) with P x Q (2M-1, 2N-1)
length = image.shape[0]
mask = np.zeros([length * 2,length * 2,3], dtype=np.uint8)
mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
image_b = mask
image_b[:length,:length] = image

# image(c): multiply image(b) by (-1)^(x+y)
image_c = np.zeros_like(image_b)
height, weight = image_b.shape[0], image_b.shape[1]
for i in range(height):
    for j in range(weight):
        image_c[i, j] = image_b[i, j] * ((-1) ** (i + j))

# 傅立葉轉換 Fp, 取得 Fp 的 spectrum (不需要 fftshift)
fp = np.fft.fft2(image_c)
spectrum = 20*np.log(1 + np.abs(fp))

# 建一個 centered gaussian lowpass filter
h_mask = np.zeros_like(image_c)
centerx, centery = h_mask.shape[0] // 2, h_mask.shape[1] // 2
cv2.circle(h_mask, (centerx,centery), 32, (255,255,255), -1)[0]
h_mask = cv2.GaussianBlur(h_mask, (1033, 1033), 5)

# 將 Fp 與 H product 後得到 HFp, 並取得其 spectrum
hfp = np.multiply(fp,h_mask) / 255.0
hfp_spectrum = 20*np.log(1 + np.abs(hfp))

# 取 HFp idft 後的 real part, multiply by (-1)^(x+y)
hfp_real = np.fft.ifft2(hfp).real
for i in range(height):
    for j in range(weight):
        hfp_real[i, j] = hfp_real[i, j] * ((-1) ** (i + j))
# crop 出 M x N 的 final result
final_img= hfp_real[:length,:length]
# 以下繪圖
plt.subplot(331),plt.imshow(image, cmap = 'gray')
plt.title('Fig 4.36(a)'), plt.xticks([]), plt.yticks([])
plt.subplot(332),plt.imshow(image_b, cmap = 'gray')
plt.title('Fig 4.36(b)'), plt.xticks([]), plt.yticks([])
plt.subplot(333),plt.imshow(image_c, cmap = 'gray')
plt.title('Fig 4.36(c)'), plt.xticks([]), plt.yticks([])
plt.subplot(334),plt.imshow(spectrum, cmap = 'gray')
plt.title('Fig 4.36(d)'), plt.xticks([]), plt.yticks([])
plt.subplot(335),plt.imshow(np.abs(h_mask), cmap = 'gray')
plt.title('Fig 4.36(e)'), plt.xticks([]), plt.yticks([])
plt.subplot(336),plt.imshow(hfp_spectrum, cmap = 'gray')
plt.title('Fig 4.36(f)'), plt.xticks([]), plt.yticks([])
plt.subplot(337),plt.imshow(hfp_real, cmap = 'gray')
plt.title('Fig 4.36(g)'), plt.xticks([]), plt.yticks([])
plt.subplot(338),plt.imshow(final_img, cmap = 'gray')
plt.title('Fig 4.36(h)'), plt.xticks([]), plt.yticks([])
plt.show()