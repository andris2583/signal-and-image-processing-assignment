import numpy as np
from skimage import data
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
from skimage import io, color
from scipy.ndimage import gaussian_filter


def one():
  image = io.imread("pout.tif") / 255.0
  
  bin_counts, bin_edges = np.histogram(image, 256)

  cdf = np.cumsum(bin_counts)
  cdf = cdf / cdf[-1]
  plt.plot(cdf, color='blue')
  plt.show()
  

def cdfTransform(image, cdf):
    image = (image * 255).astype(np.uint8)

    transformed_image = cdf[image]

    return transformed_image

def two():
  image = io.imread("pout.tif") / 255.0
  
  histogram, _ = np.histogram(image, bins=256, range=[0,1])

  cdf = np.cumsum(histogram)
  cdf = cdf / cdf[-1]

  transformed_image = cdfTransform(image, cdf)

  fig, ax = plt.subplots(1, 2, figsize=(12, 5))
  ax[0].imshow(image, cmap="gray")
  ax[0].set_title("Original Image")
  ax[0].axis("off")

  ax[1].imshow(transformed_image, cmap="gray")
  ax[1].set_title("Transformed using CDF")
  ax[1].axis("off")

  plt.show()
  
def pseudo_inverse(cdf):

    cdf = cdf / cdf[-1]
    
    def calculate_pseudo_inverse(l):
        return np.min(np.where(cdf >= l))
    
    l_values = np.linspace(0, 1, num=256)
    pseudo_inverse_values = np.array([calculate_pseudo_inverse(l) for l in l_values])
    
    return pseudo_inverse_values

def compute_cdf(image):

    histogram, _ = np.histogram(image, bins=256, range=[0,1])
    cdf = np.cumsum(histogram)

    return cdf / cdf[-1]

def histogram_matching(source_image, target_image):
    source_cdf = compute_cdf(source_image)  
    target_cdf = compute_cdf(target_image)  

    transformed_source = (cdfTransform(source_image, source_cdf)* 255).astype(np.uint8)

    pseudo_inverse_target = pseudo_inverse(target_cdf)

    matched_image = pseudo_inverse_target[transformed_source]

    return matched_image

def four():
  source_image = io.imread("mandi.tif") / 255.0
  target_image = io.imread("hand.tiff") / 255.0

  matched_image = histogram_matching(source_image, target_image)

  # Plot the results of 4.4
  fig, ax = plt.subplots(1, 3, figsize=(15, 5))
  ax[0].imshow(source_image, cmap="gray")
  ax[0].set_title("Source Image")
  ax[0].axis("off")

  ax[1].imshow(target_image, cmap="gray")
  ax[1].set_title("Target Image")
  ax[1].axis("off")

  ax[2].imshow(matched_image, cmap="gray")
  ax[2].set_title("Histogram Matched Image")
  ax[2].axis("off")

  plt.show()

  source_hist, bins = np.histogram(source_image.flatten(), bins=256, range=[0,1], density=True)
  target_hist, _ = np.histogram(target_image.flatten(), bins=256, range=[0,1], density=True)
  matched_hist, _ = np.histogram(matched_image.flatten(), bins=256, range=[0,255], density=True)


  plt.figure(figsize=(15, 5))
  plt.subplot(1, 3, 1)
  plt.bar(bins[:-1], source_hist, width=0.004, color='blue', alpha=0.7)
  plt.title("Source Image Histogram")
  plt.xlabel("Pixel Intensity")
  plt.ylabel("Density")

  plt.subplot(1, 3, 2)
  plt.bar(bins[:-1], target_hist, width=0.004, color='green', alpha=0.7)
  plt.title("Target Image Histogram")
  plt.xlabel("Pixel Intensity")

  plt.subplot(1, 3, 3)
  plt.bar(bins[:-1], matched_hist, width=0.004, color='red', alpha=0.7)
  plt.title("Matched Image Histogram")
  plt.xlabel("Pixel Intensity")

  plt.tight_layout()
  plt.show()

two()