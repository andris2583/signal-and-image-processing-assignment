import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, show, subplot, title
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import median, gaussian
from skimage.filters.rank import mean
from skimage.morphology import disk
from skimage.util import random_noise
import timeit

image = img_as_float(io.imread('eight.tif', as_gray=True))

def one():
    
    salt_pepper_noise = img_as_ubyte(random_noise(image, mode='s&p', amount=0.1))
    gaussian_noise = img_as_ubyte(random_noise(image, mode='gaussian', var=0.01))


    def mean_filter(image, N):
        return mean(image, disk(N))

    def median_filter(image, N):
        return median(image, disk(N))
    
    kernel_size = 25
    mean_filtered_sp = mean_filter(salt_pepper_noise, kernel_size)
    median_filtered_sp = median_filter(salt_pepper_noise, kernel_size)
    mean_filtered_gaussian = mean_filter(gaussian_noise, kernel_size)
    median_filtered_gaussian = median_filter(gaussian_noise, kernel_size)

    # salt & pepper results:
    subplot(2, 2, 1)
    title('Original')
    imshow(image, cmap='gray')

    subplot(2, 2, 2)
    title('Salt & Pepper Noise')
    imshow(salt_pepper_noise, cmap='gray')

    subplot(2, 2, 3)
    title('Mean Filter (Salt & Pepper Noise)')
    imshow(mean_filtered_sp, cmap='gray')

    subplot(2, 2, 4)
    title('Median Filter (Salt & Pepper Noise)')
    imshow(median_filtered_sp, cmap='gray')

    show()

    #Gaussian noise results:
    subplot(2, 2, 1)
    title('Original')
    imshow(image, cmap='gray')

    subplot(2, 2, 2)
    title('Gaussian Noise')
    imshow(gaussian_noise, cmap='gray')

    subplot(2, 2, 3)
    title('Mean Filter (Gaussian Noise)')
    imshow(mean_filtered_gaussian, cmap='gray')

    subplot(2, 2, 4)
    title('Median Filter (Gaussian Noise)')
    imshow(median_filtered_gaussian, cmap='gray')

    show()


    #kernel_sizes = range(1, 26)

    #mean_times = []
    #median_times = []

    #for N in kernel_sizes:
    #    mean_time = timeit.timeit(lambda: mean_filter(salt_pepper_noise, N), number=100)
    #    median_time = timeit.timeit(lambda: median_filter(salt_pepper_noise, N), number=100)
    #    mean_times.append(mean_time)
    #    median_times.append(median_time)

    # Plotting the results
    #plt.plot(kernel_sizes, mean_times, label='Mean Filter')
    #plt.plot(kernel_sizes, median_times, label='Median Filter')
    #plt.xlabel('Kernel Size')
    #plt.ylabel('Computation Time (s)')
    #plt.legend()
    #plt.show()


def two():
    sigma = 5

    kernel_sizes = [3, 10, 15, 20, 25]

    filtered_images = [gaussian(image, sigma=sigma, truncate=ks / (2 * sigma)) for ks in kernel_sizes]

    
    subplot(2, 3, 1)
    title('Original Image')
    imshow(image, cmap='gray')

    for i, ks in enumerate(kernel_sizes):
        subplot(2, 3, i + 2)
        title(f'Gaussian Filter (Kernel Size = {ks})')
        imshow(filtered_images[i], cmap='gray')

    show()
    
def three():
    sigmas = [1, 2, 3, 4, 5]
    kernel_sizes = [int(3*sigma) for sigma in sigmas]

    filtered_images = [gaussian(image, sigma=sigma, truncate=3) for sigma in sigmas]

    subplot(2, 3, 1)
    title('Original Image')
    imshow(image, cmap='gray')

    for i, sigma in enumerate(sigmas):
        subplot(2, 3, i + 2)
        title(f'Gaussian Filter (σ = {sigma}, N = {kernel_sizes[i]})')
        imshow(filtered_images[i], cmap='gray')

    show()
    
#one()
#two()
#three()

    