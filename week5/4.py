import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter, laplace
from skimage.feature import peak_local_max
from skimage.filters import gaussian

def one():
    rows, cols = 10, 10
    gaussian = np.zeros((rows,cols))
    x_center,y_center  = cols//2, rows//2
    sigma = 1
    for x in range(rows):
      for y in range(cols):
        gaussian_img[x,y] = np.e**(-((x - x_center)**2 + (y - y_center)**2) / (2*sigma**2))
    gaussian_img = 1/(2*np.pi * sigma**2) * gaussian_img

    tau_img = np.zeros((rows,cols))
    tau = 2
    for x in range(rows):
      for y in range(cols):
        tau_img[x,y] = np.e**(-((x - x_center)**2 + (y - y_center)**2) / (2*tau**2))
    tau_img = 1/(2*np.pi * tau**2) * tau_img

    convolved = gaussian_img * tau_img

    difference = tau_img - gaussian_img

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(gaussian_img, cmap='gray'); ax[0].set_title("Original Gaussian (σ=1)")
    ax[1].imshow(tau_img, cmap='gray'); ax[1].set_title("Original Tau (τ=2)")
    ax[2].imshow(convolved, cmap='gray'); ax[2].set_title("Gaussian convolved with Tau")
    ax[3].imshow(difference, cmap='gray'); ax[3].set_title("Difference")
    plt.show()

def three():
  sigma = 1
  tau_vals = np.linspace(0, 5, 100)
  H_vals = - (2 * tau_vals**2) / (sigma**2 + tau_vals**2)

  plt.figure(figsize=(8, 5))
  plt.plot(tau_vals, H_vals, label=r'$H(0,0,\tau) = -\frac{2\tau^2}{\sigma^2+\tau^2}$')
  plt.xlabel(r'$\tau$')
  plt.ylabel(r'$H(0,0,\tau)$')
  plt.title('Scale-normalized Laplacian at (0,0) as a function of scale')
  plt.legend()
  plt.grid()
  plt.show()

def laplacian_of_gaussian(image, sigma):
    return sigma**2 * laplace(gaussian(image, sigma=sigma))

def four():
    image = io.imread('/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 5/Assignment/sunflower.tiff', as_gray=True)

    sigma_values = np.linspace(1, 10, 10)
    scale_space = np.zeros((image.shape[0], image.shape[1], len(sigma_values)))

    for i, sigma in enumerate(sigma_values):
        scale_space[:, :, i] = laplacian_of_gaussian(image, sigma)

    maxima = peak_local_max(scale_space, num_peaks=150, threshold_abs=0.01, footprint=np.ones((3,3,3)))
    minima = peak_local_max(-scale_space, num_peaks=150, threshold_abs=0.01, footprint=np.ones((3,3,3)))

    y_max, x_max, scale_max = maxima.T
    y_min, x_min, scale_min = minima.T
    detected_scales_max = sigma_values[scale_max]
    detected_scales_min = sigma_values[scale_min]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    max_patches = []
    for x, y, s in zip(x_max, y_max, detected_scales_max):
        circle = plt.Circle((x, y), s, color='red', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        max_patches.append(circle)

    min_patches = []
    for x, y, s in zip(x_min, y_min, detected_scales_min):
        circle = plt.Circle((x, y), s, color='blue', fill=False, linewidth=1.5)
        ax.add_patch(circle)
        min_patches.append(circle)

    max_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Maxima (Bright Blobs)')
    min_legend = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Minima (Dark Blobs)')

    plt.legend(handles=[max_legend, min_legend], loc="lower right")

    plt.title("Blob Detection with Scale-Space")
    plt.show()

    return 0

four()