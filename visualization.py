import cv2
import numpy as np
import os

PHOTO = 'data/tune/images/207468.png'

img_bgr = cv2.imread(PHOTO)
if img_bgr is None:
    print(f"Ошибка: не удалось загрузить {PHOTO}")
    exit()

print(f'Размер (H, W, C): {img_bgr.shape}')
print(f'Тип данных: {img_bgr.dtype}')
print(f'Минимальное значение: {img_bgr.min()}')
print(f'Максимальное значение: {img_bgr.max()}')

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

max_width = 600

def resize_to_width(img, width):
    h = int(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h))

img_bgr_resized = resize_to_width(img_bgr, max_width)
img_rgb_resized = resize_to_width(img_rgb, max_width)
img_hsv_resized = resize_to_width(img_hsv, max_width)

result = np.hstack((img_bgr_resized, img_rgb_resized, img_hsv_resized))
cv2.imshow('BGR | RGB | HSV', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('result.png', img_rgb)
cv2.imwrite('result.jpg', img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

print(f"PNG: {os.path.getsize('result.png')} байт")
print(f"JPEG: {os.path.getsize('result.jpg')} байт")