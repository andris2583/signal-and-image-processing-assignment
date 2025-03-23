import matplotlib.pyplot as plt
from skimage import io, morphology, measure, color
from skimage.color import rgb2gray
import numpy as np


img = io.imread('cells_binary_inv.png')
if img.ndim == 3:
    img = rgb2gray(img)
img = (img > 0.5).astype(np.uint8)  

selem = morphology.disk(1)

opened = morphology.opening(img, selem)
closed = morphology.closing(img, selem)



zoom1 = (250, 350, 350, 450)  
zoom2 = (175, 275, 350, 450)  

def show_images_double_zoom(original, opened, closed, zoom1, zoom2):
    y1b, y1e, x1b, x1e = zoom1
    y2b, y2e, x2b, x2e = zoom2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    ax = axes.ravel()

    ax[0].imshow(original, cmap='gray')
    ax[0].set_title("Original Image")
    ax[1].imshow(original[y1b:y1e, x1b:x1e], cmap='gray')
    ax[1].set_title("Zoom-in (Original)")
    ax[2].imshow(original[y2b:y2e, x2b:x2e], cmap='gray')
    ax[2].set_title("Zoom-in (Original)")

    ax[3].imshow(opened, cmap='gray')
    ax[3].set_title("Opening")
    ax[4].imshow(opened[y1b:y1e, x1b:x1e], cmap='gray')
    ax[4].set_title("Zoom-in (Opened)")
    ax[5].imshow(opened[y2b:y2e, x2b:x2e], cmap='gray')
    ax[5].set_title("Zoom-in (Opened)")

    ax[6].imshow(closed, cmap='gray')
    ax[6].set_title("Closing")
    ax[7].imshow(closed[y1b:y1e, x1b:x1e], cmap='gray')
    ax[7].set_title("Zoom-in (Closed)")
    ax[8].imshow(closed[y2b:y2e, x2b:x2e], cmap='gray')
    ax[8].set_title("Zoom-in (Closed)")

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

#show_images_double_zoom(img, opened, closed, zoom1, zoom2)

def components():
    labels_opened = measure.label(opened, connectivity=2)
    labels_closed = measure.label(closed, connectivity=2)

    num_components_opened = np.max(labels_opened)
    num_components_closed = np.max(labels_closed)

    print(f"Number of connected components after opening: {num_components_opened}")
    print(f"Number of connected components after closing: {num_components_closed}")

    labeled_image_opened = color.label2rgb(labels_opened, bg_label=0)
    labeled_image_closed = color.label2rgb(labels_closed, bg_label=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(labeled_image_opened)
    axes[0].set_title(f"Connected components after opening ({num_components_opened} components)")
    axes[0].axis('off')

    axes[1].imshow(labeled_image_closed)
    axes[1].set_title(f"Connected components after closing ({num_components_closed} components)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

#components()

def three():
    img = io.imread('saved_image.png', as_gray=True)
   
    selem = morphology.disk(6)  
    cleaned_img = morphology.closing(img, selem)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Binary Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_img, cmap='gray')
    plt.title('Cleaned Binary Image (Closing)')
    plt.axis('off')

    plt.show()
    
three()
    
    