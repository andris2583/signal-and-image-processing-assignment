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

two()