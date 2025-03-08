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
  H_vals = (tau_vals**2) * (2/(sigma**2+tau_vals**2)**4 - 2/(sigma**2+tau_vals**2))

  plt.figure(figsize=(8, 5))
  plt.plot(tau_vals, H_vals)
  plt.xlabel(r'$\tau$')
  plt.ylabel(r'$H(0,0,\tau)$')
  plt.title('Scale-normalized Laplacian at (0,0) as a function of scale')
  plt.legend()
  plt.grid()
  plt.show()

def laplacian_of_gaussian(image, sigma):
    return sigma**2 * laplace(gaussian(image, sigma=sigma))

def four():
    image = io.imread('sunflower.tiff', as_gray=True)

    sigma_values = np.linspace(1, 10, 10)
    scale_space = np.zeros((image.shape[0], image.shape[1], len(sigma_values)))

    for i, sigma in enumerate(sigma_values):
        scale_space[:, :, i] = laplacian_of_gaussian(image, sigma)

    blobs = peak_local_max(np.abs(scale_space), num_peaks=150, threshold_abs=0.01, footprint=np.ones((3, 3, 3)))

    y_coords, x_coords, scale_indices = blobs.T
    detected_scales = sigma_values[scale_indices]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')

    for x, y, s in zip(x_coords, y_coords, detected_scales):
        circle = plt.Circle((x, y), s, color='red', fill=False, linewidth=1.5)
        ax.add_patch(circle)

    plt.title("Blob Detection with Scale-Space")
    plt.show()
    
    return blobs

four()