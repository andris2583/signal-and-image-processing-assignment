import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.signal import convolve2d

def apply_lsi_degradation(image, kernel, noise):
    blurred_image = convolve2d(image, kernel, mode='same', boundary='wrap')
    degraded_image = blurred_image + noise
    
    return degraded_image

image = img_as_float(io.imread('trui.png', as_gray=True))

kernel1 = np.ones((3, 3)) / 9  
kernel2 = np.array([[2, 4, 2], 
                        [5, 15, 5], 
                        [2, 4, 2]]) / 16  
kernel3 = np.array([[3, 1, 0, 1, 3], 
                        [1, 3, 0, 0, 0], 
                        [0, 1, 3, 1, 0], 
                        [0, 0, 0, 3, 1], 
                        [0, 0, 0, 0, 3]]) / 5  

noise_levels = [0.02, 0.05, 0.1]

fig, axes = plt.subplots(len(noise_levels), 4, figsize=(15, 10))

for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(scale=noise_level, size=image.shape)
    
    deg_ex_1 = apply_lsi_degradation(image, kernel1, noise)
    deg_ex_2 = apply_lsi_degradation(image, kernel2, noise)
    deg_ex_3 = apply_lsi_degradation(image, kernel3, noise)
    

    axes[i, 0].imshow(image, cmap='gray')
    axes[i, 0].set_title(f"Original Image")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(deg_ex_1, cmap='gray')
    axes[i, 1].set_title(f"Kernel example 1 + Noise {noise_level}")
    axes[i, 1].axis("off")
    
    axes[i, 2].imshow(deg_ex_2, cmap='gray')
    axes[i, 2].set_title(f"Kernel example 2 + Noise {noise_level}")
    axes[i, 2].axis("off")
    
    axes[i, 3].imshow(deg_ex_3, cmap='gray')
    axes[i, 3].set_title(f"Kernel example 3 + Noise {noise_level}")
    axes[i, 3].axis("off")

#plt.tight_layout()
#plt.show()

'''def inverse_filtering(degraded_image, kernel, epsilon=1e-3):
    
    kernel_padded = np.zeros_like(degraded_image)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    G = np.fft.fft2(degraded_image)
    H = np.fft.fft2(kernel_padded)
    
    H_inv = np.conj(H) / (np.abs(H) ** 2 + epsilon)
    
    F_hat = G * H_inv
    
    restored_image = np.fft.ifft2(F_hat).real
    
    return np.clip(restored_image, 0, 1)  

kernel = np.array([[1, 2, 1], 
                   [2, 4, 2], 
                   [1, 2, 1]]) / 16

noise_levels = [0.01, 0.05, 0.2]  

fig, axes = plt.subplots(len(noise_levels), 3, figsize=(15, 10))

for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(scale=noise_level, size=image.shape)
    
    degraded_image = apply_lsi_degradation(image, kernel, noise)
    
    restored_image = inverse_filtering(degraded_image, kernel, epsilon=1e-2)

    axes[i, 0].imshow(image, cmap='gray')
    axes[i, 0].set_title("Original Image")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(degraded_image, cmap='gray')
    axes[i, 1].set_title(f"Degraded Image (Noise {noise_level})")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(restored_image, cmap='gray')
    axes[i, 2].set_title(f"Restored Image")
    axes[i, 2].axis("off")

    

plt.tight_layout()
plt.show()'''

def wiener_filtering(degraded_image, kernel, K=0.1, epsilon=1e-3):

    kernel_padded = np.zeros_like(degraded_image)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel

    G = np.fft.fft2(degraded_image)
    H = np.fft.fft2(kernel_padded)

    G_mag2 = np.abs(G) ** 2
    H_mag2 = np.abs(H) ** 2

    noise_power = np.mean(G_mag2)  
    H_wiener = np.conj(H) / (H_mag2 + K * noise_power)
    F_hat = G * H_wiener

    restored_image = np.fft.ifft2(F_hat).real

    return np.clip(restored_image, 0, 1)  

kernel = np.array([[1, 2, 1], 
                   [2, 4, 2], 
                   [1, 2, 1]]) / 16


noise_levels = [0.01, 0.05, 0.2]  
K_values = [0.01, 0.1, 0.5]  

fig, axes = plt.subplots(len(noise_levels), len(K_values)+1, figsize=(15, 10))

for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(scale=noise_level, size=image.shape)
    
    degraded_image = apply_lsi_degradation(image, kernel, noise)
    
    for j, K in enumerate(K_values):
        restored_image = wiener_filtering(degraded_image, kernel, K=K)
        
        axes[i, j+1].imshow(restored_image, cmap='gray')
        axes[i, j+1].set_title(f"K={K}, Noise={noise_level}")
        axes[i, j+1].axis("off")
    
    axes[i, 0].imshow(degraded_image, cmap='gray')
    axes[i, 0].set_title(f"Degraded Image (Noise {noise_level})")
    axes[i, 0].axis("off")

plt.tight_layout()
plt.show()