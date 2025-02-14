import numpy as np
from skimage import data
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import gaussian_filter


def gammaCorrection(image, gamma, c):
    image = np.clip(image, 0, 1)
    corrected = c * np.power(image, gamma)
    return np.clip(corrected, 0, 1)

def one():
  image = io.imread("modelhouses.png") / 255.0

  gamma_image = gammaCorrection(image, 0.5, 1)

  fig, ax = plt.subplots(1, 3, figsize=(10, 5))
  ax[0].imshow(image, cmap="gray")
  ax[0].set_title("Grayscale image")
  ax[0].axis("off")

  ax[1].imshow(gamma_image, cmap="gray")
  ax[1].set_title("Gamma = 0.5")
  ax[1].axis("off")
  
  gamma_image_2 = gammaCorrection(image, 2, 1)
  ax[2].imshow(gamma_image_2, cmap="gray")
  ax[2].set_title("Gamma = 2")
  ax[2].axis("off")

  plt.show()


def two():
  image = io.imread("autumn.tif") / 255.0
  red = image[:, :, 0]
  green = image[:, :, 1]
  blue = image[:, :, 2]
  
  red_gamma = gammaCorrection(red, 0.8, 1)
  green_gamma = gammaCorrection(green, 0.8, 1)
  blue_gamma = gammaCorrection(blue, 0.8, 1)

  gamma_image = np.dstack((red_gamma,green_gamma, blue_gamma))
  print(gamma_image)
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].imshow(image, cmap="gray")
  ax[0].set_title("Color image")
  ax[0].axis("off")

  ax[1].imshow(gamma_image, cmap="gray")
  ax[1].set_title("Gamma corrected color image")
  ax[1].axis("off")

  plt.show()

def three():
  image = io.imread("autumn.tif") / 255.0
  
  hsv_image = rgb2hsv(image)

  value = hsv_image[:, :, 2]

  value_gamma = gammaCorrection(value, 0.8, 1)

  gamma_hsv_image = np.dstack((hsv_image[:, :, 0],hsv_image[:, :, 1], value_gamma))
  gamma_rgb_image = hsv2rgb(gamma_hsv_image)

  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
  ax[0].imshow(image, cmap="gray")
  ax[0].set_title("Color image")
  ax[0].axis("off")

  ax[1].imshow(gamma_rgb_image, cmap="gray")
  ax[1].set_title("Gamma corrected color image")
  ax[1].axis("off")

  plt.show()

# one()
# two()
three()