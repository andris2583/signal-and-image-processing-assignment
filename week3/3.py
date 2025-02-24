"""
### 3.1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.io import imread

test_image = imread("trui.png")

fft_result_large = fft2(test_image)
fft_shifted_large = fftshift(fft_result_large)
power_spectrum_large = np.abs(fft_shifted_large) ** 2

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].imshow(test_image, cmap='gray')
ax[0].set_title("Trui Image")

ax[1].imshow(np.log1p(power_spectrum_large), cmap='grey')
ax[1].set_title("Power Spectrum (Log Scale)")

plt.show()

"""### 3.2"""

image = imread('cameraman.tif', as_gray=True)
image = image / np.max(image)

a0 = 0.3
v0, w0 = 15, 10

x = np.arange(image.shape[1])
y = np.arange(image.shape[0])
X, Y = np.meshgrid(x, y)
sinusoidal_pattern = a0 * np.cos(2 * np.pi * (v0 * X / image.shape[1] + w0 * Y / image.shape[0]))

noisy_image = image + sinusoidal_pattern
noisy_image = np.clip(noisy_image, 0, 1)

fft_noisy = fft2(noisy_image)
fft_noisy_shifted = fftshift(fft_noisy)
power_spectrum_noisy = np.abs(fft_noisy_shifted) ** 2

fft_original = fft2(image)
fft_original_shifted = fftshift(fft_original)
power_spectrum_original = np.abs(fft_original_shifted) ** 2

notch_filter = np.ones_like(fft_noisy_shifted)
notch_filter[image.shape[0]//2 + w0, image.shape[1]//2 + v0] = 0
notch_filter[image.shape[0]//2 - w0, image.shape[1]//2 - v0] = 0

fft_filtered = fft_noisy_shifted * notch_filter
ifft_filtered = ifft2(ifftshift(fft_filtered))
filtered_image = np.abs(ifft_filtered)

power_spectrum_filtered = np.abs(fft_filtered) ** 2


fig, axes = plt.subplots(4, 2, figsize=(30, 30))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(np.log1p(power_spectrum_original), cmap='grey_r')
axes[0, 1].set_title("Power Spectrum (Original Image)")

axes[1, 0].imshow(sinusoidal_pattern, cmap='gray')
axes[1, 0].set_title("Sinusoidal Pattern")
axes[1, 1].imshow(np.abs(notch_filter), cmap='grey_r')
axes[1, 1].set_title("Notch Filter (Zeros at Noise Frequencies)")

axes[2, 0].imshow(noisy_image, cmap='gray')
axes[2, 0].set_title("Noisy Image (Image + Sinusoid)")
axes[2, 1].imshow(np.log1p(power_spectrum_noisy), cmap='grey_r')
axes[2, 1].set_title("Power Spectrum (Noisy Image)")

axes[3, 0].imshow(filtered_image, cmap='gray')
axes[3, 0].set_title("Filtered Image (Sinusoid Removed)")
axes[3, 1].imshow(np.log1p(power_spectrum_filtered), cmap='grey_r')
axes[3, 1].set_title("Power Spectrum (Filtered Image)")

plt.tight_layout()
plt.show()

"""### 3.3"""

def fourier_derivative(image, dx=0, dy=0):
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)

    M, N = image.shape
    u = np.fft.fftfreq(N) * N
    v = np.fft.fftfreq(M) * M
    U, V = np.meshgrid(u, v)

    derivative_kernel = (2j * np.pi * U) ** dx * (2j * np.pi * V) ** dy
    derivative_kernel = np.fft.ifftshift(derivative_kernel)

    fft_derivative = fft_image_shifted * derivative_kernel

    derivative_image = np.abs(ifft2(ifftshift(fft_derivative)))

    return derivative_image

test_image = imread('bigben.png', as_gray=True)
original_image = imread('bigben.png')

dx1 = fourier_derivative(test_image, dx=1, dy=0)
dy1 = fourier_derivative(test_image, dx=0, dy=1)
dx2 = fourier_derivative(test_image, dx=2, dy=0)
dy2 = fourier_derivative(test_image, dx=0, dy=2)
dx3 = fourier_derivative(test_image, dx=10, dy=0)
dy3 = fourier_derivative(test_image, dx=0, dy=10)

fig, axes = plt.subplots(4, 2, figsize=(30, 30))

im1 = axes[0, 0].imshow(dx1, cmap='grey')
axes[0, 0].set_title("First Derivative (X)")
axes[0, 0].axis("off")
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(dy1, cmap='grey')
axes[0, 1].set_title("First Derivative (Y)")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[1, 0].imshow(dx2, cmap='grey')
axes[1, 0].set_title("Second Derivative (X)")
axes[1, 0].axis("off")
plt.colorbar(im3, ax=axes[1, 0])

im4 = axes[1, 1].imshow(dy2, cmap='grey')
axes[1, 1].set_title("Second Derivative (Y)")
axes[1, 1].axis("off")
plt.colorbar(im4, ax=axes[1, 1])

im3 = axes[2, 0].imshow(dx2, cmap='grey')
axes[2, 0].set_title("10th Derivative (X)")
axes[2, 0].axis("off")
plt.colorbar(im3, ax=axes[2, 0])

im4 = axes[2, 1].imshow(dy2, cmap='grey')
axes[2, 1].set_title("10th Derivative (Y)")
axes[2, 1].axis("off")
plt.colorbar(im4, ax=axes[2, 1])

axes[3, 0].imshow(original_image)
axes[3, 0].set_title("Original Image")
axes[3, 0].axis("off")

axes[3, 1].imshow(test_image, cmap='gray')
axes[3, 1].set_title("Original Gray Scale Image")
axes[3, 1].axis("off")
plt.show()