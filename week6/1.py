import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float


def create_centered_white_square(image_size, square_size):

    img = np.zeros(image_size, dtype=np.uint8)

    cx, cy = image_size[1] // 2, image_size[0] // 2
    sx, sy = square_size[1] // 2, square_size[0] // 2
    
    img[cy - sy:cy + sy + 1, cx - sx:cx + sx + 1] = 255
    
    return img

def translate_image(image, tx, ty):

    translated_image = np.zeros_like(image, dtype=np.uint8)

    translated_image[max(0, ty):min(image.shape[0], image.shape[0] + ty),
                     max(0, tx):min(image.shape[1], image.shape[1] + tx)] = image[max(0, -ty):min(image.shape[0], image.shape[0] - ty),
                                                                                   max(0, -tx):min(image.shape[1], image.shape[1] - tx)]
    
    return translated_image

def translate_image_non_int(image, t):
    tx, ty = t  
    height, width = image.shape
    translated_image_2 = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            x = j - tx
            y = i - ty
            
            x_rounded = int(round(x))
            y_rounded = int(round(y))

            if 0 <= x_rounded < width and 0 <= y_rounded < height:
                translated_image_2[i, j] = image[y_rounded, x_rounded]

    return translated_image_2

def translate_image_fourier(image, t):
    tx, ty = t
    
    F = np.fft.fftshift(np.fft.fft2(image))
    
    rows, cols = image.shape
    u = np.fft.fftfreq(cols, 1)  
    v = np.fft.fftfreq(rows, 1)  
    u, v = np.meshgrid(u, v)
    
    F_translated = F * np.exp(-2j * np.pi * (u * tx + v * ty))
    
    translated_image_fourier = np.abs(np.fft.ifft2(np.fft.ifftshift(F_translated)))
    
    translated_image_fourier = np.clip(translated_image_fourier, 0, 255).astype(np.uint8)
    
    return translated_image_fourier

image_size = (101, 101)  
square_size = (25, 25)   
img = create_centered_white_square(image_size, square_size)

t = (0.6, 1.2)

translated_img = translate_image(img, tx=15, ty=-15)
translated_img_2 = translate_image_non_int(img, t)
translated_img_fourier = translate_image_fourier(img, t)

other_image = img_as_float(io.imread('toycars2.png', as_gray=True))
translated_img_fourier_2 = translate_image_fourier(other_image, t)

#fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#ax[0].imshow(img, cmap='gray')
#ax[0].set_title("Original Image")
#ax[0].axis('off')

#ax[1].imshow(translated_img, cmap='gray')
#ax[1].set_title("Translated Image (15px right, 15px up)")
#ax[1].axis('off')

#ax[1].imshow(translated_img_2, cmap='gray')
#ax[1].set_title(f"Translated Image (t = {t})")
#ax[1].axis('off')

#ax[1].imshow(translated_img_fourier, cmap='gray')
#ax[1].set_title(f"Translated Image (Fourier, t = {t})")
#ax[1].axis('off')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(other_image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis('off')

ax[1].imshow(translated_img_fourier_2, cmap='gray')
ax[1].set_title(f"Translated Image (Fourier, t = {t})")
ax[1].axis('off')

plt.show()
