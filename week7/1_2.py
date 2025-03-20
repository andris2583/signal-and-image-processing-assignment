# -*- coding: utf-8 -*-
"""
## Section 1
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import color, filters, feature, transform

"""### 1.1"""

image_paths = [
    "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 7/Assignment/matrikelnumre_nat.png",
    "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 7/Assignment/matrikelnumre_art.png"
]
image_names = ["matrikelnumre_nat", "matrikelnumre_art"]

for img_path, img_name in zip(image_paths, image_names):
    img_color = imread(img_path)
    img_gray = rgb2gray(img_color)

    hist_values, bin_edges = np.histogram(img_gray.ravel(), bins=256, range=(0, 1))

    otsu_thresh = threshold_otsu(img_gray)
    binary_seg = img_gray > otsu_thresh

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(img_color)
    axes[0].set_title(f"Original ({img_name})")
    axes[0].axis("off")

    axes[1].imshow(img_gray, cmap="gray")
    axes[1].set_title(f"Grayscale ({img_name})")
    axes[1].axis("off")

    axes[2].hist(img_gray.ravel(), bins=256, range=(0, 1), color="black")
    axes[2].axvline(otsu_thresh, color='red', linestyle='dashed', label=f'Otsu Threshold = {otsu_thresh:.3f}')
    axes[2].set_title(f"Histogram ({img_name})")
    axes[2].legend()

    axes[3].imshow(binary_seg, cmap="gray")
    axes[3].set_title(f"Segmented ({img_name})")
    axes[3].axis("off")

    plt.suptitle(f"Otsu Thresholding Results for {img_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{img_name}_otsu.png")
    plt.show()

"""### 1.2"""

# Load the image in grayscale
image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 7/Assignment/matrikelnumre_art.png"
img_gray = imread(image_path, as_gray=True)

# Apply Canny Edge Detection
edges = feature.canny(img_gray, sigma=1.5)  # Sigma controls edge sensitivity

# Plot the results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(img_gray, cmap="gray")
ax[0].set_title("Grayscale Image")
ax[0].axis("off")

ax[1].imshow(edges, cmap="gray")
ax[1].set_title("Canny Edge Detection")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("matrikelnumre_art_canny.png")
plt.show()

"""## Section 2

### 2.1
"""

def find_map_corners(image_path, num_peaks=50, border_margin=10):
    image = io.imread(image_path)
    gray = color.rgb2gray(image)

    corner_response = feature.corner_harris(gray, method='eps', sigma=1, eps=1)

    detected_corners = feature.corner_peaks(corner_response, num_peaks=num_peaks, threshold_rel=0.01)

    img_height, img_width = gray.shape

    valid_corners = detected_corners[
        (detected_corners[:, 0] > border_margin) & (detected_corners[:, 0] < img_height - border_margin) &
        (detected_corners[:, 1] > border_margin) & (detected_corners[:, 1] < img_width - border_margin)
    ]

    top_left = valid_corners[np.argmin(np.sum(valid_corners, axis=1))]
    top_right = valid_corners[np.argmin(valid_corners[:, 0] - valid_corners[:, 1])]
    bottom_left = valid_corners[np.argmax(valid_corners[:, 0] - valid_corners[:, 1])]
    bottom_right = valid_corners[np.argmax(np.sum(valid_corners, axis=1))]

    map_corners = np.array([top_left, top_right, bottom_left, bottom_right])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(gray, cmap="gray")
    ax.scatter(valid_corners[:, 1], valid_corners[:, 0], s=30, c='yellow', label="Top 50 Detected Corners")
    ax.scatter(map_corners[:, 1], map_corners[:, 0], s=100, c='red', edgecolors='black', label="Final 4 Corners")
    ax.set_title("Filtered Corners of the Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("corners.png")
    plt.show()

    return map_corners

image_path ="/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 7/Assignment/matrikelnumre_nat.png"
map_corners = find_map_corners(image_path)
print("Detected Map Corners (x, y):\n", map_corners)

"""### 2.2"""

image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 7/Assignment/matrikelnumre_nat.png"
image = imread(image_path)

src = np.array([
    [301, 387],
    [1012, 408],
    [165, 1359],
    [781, 1612]]
, dtype=np.float32)

src = src[:, [1, 0]]

output_width = 2040
output_height = 1148

dst = np.array([
    [0, 0],
    [0, output_height],
    [output_width, 0],
    [output_width, output_height]
], dtype=np.float32)

tform = transform.estimate_transform('projective', src, dst)

tf_img = transform.warp(image, tform.inverse, output_shape=(output_height, output_width))

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

ax[0].imshow(image)
ax[0].scatter(src[:, 0], src[:, 1], c='red', s=50, label="Detected Corners")
ax[0].set_title("Original Image with Detected Corners")
ax[0].axis("off")

ax[1].imshow(tf_img)
ax[1].set_title("Perspective Corrected Map")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("perspective_corrected.png")
plt.show()

