# -*- coding: utf-8 -*-
"""
## Section 3
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.signal import convolve2d
import math
from sklearn.cluster import KMeans

"""### 3.1"""

image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 6/Assignment/sunandsea.jpg"
gray_image = io.imread(image_path, as_gray=True)

sigma = 5

filter_orders = {
    "G_x": ((1, 0), 0.5),
    "G_y": ((0, 1), 0.5),
    "G_xx": ((2, 0), 1),
    "G_yy": ((0, 2), 1),
    "G_xy": ((1, 1), 1),
    "G_xxx": ((3, 0), 1.5),
    "G_xxy": ((2, 1), 1.5),
    "G_xyy": ((1, 2), 1.5),
    "G_yyy": ((0, 3), 1.5),
}

filter_responses = {name: (sigma ** gamma) * gaussian_filter(gray_image, sigma=sigma, order=order)
                    for name, (order, gamma) in filter_orders.items()}

def plot_filter_responses(image, filter_responses, sigma):
    filter_names = list(filter_responses.keys())
    num_filters = len(filter_names)

    num_cols = (num_filters + 1) // 2
    fig, axes = plt.subplots(2, num_cols, figsize=(16, 10))

    axes = axes.flatten()

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Original Image (Sigma={sigma})")
    axes[0].axis("off")

    for i, (filter_name, response) in enumerate(filter_responses.items()):
        axes[i + 1].imshow(response, cmap="gray")
        axes[i + 1].set_title(f"{filter_name}, Sigma={sigma}")
        axes[i + 1].axis("off")

    for j in range(i + 2, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Scale-Normalized N-Jet Filter Bank Responses (Sigma=5)", fontsize=16)
    plt.tight_layout()
    plt.savefig("3.1.png")
    plt.show()

plot_filter_responses(gray_image, filter_responses, sigma)

"""### 3.2"""

image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 6/Assignment/sunandsea.jpg"
gray_image = io.imread(image_path, as_gray=True)

patch_size = (18, 18)
num_patches = 10000
patches = extract_patches_2d(gray_image, patch_size, max_patches=num_patches)
patches = patches.reshape(num_patches, -1)

num_components = 9
pca = PCA(n_components=num_components)
pca.fit(patches)
filters = pca.components_.reshape(num_components, *patch_size)

filtered_responses = [convolve2d(gray_image, f, mode="same") for f in filters]

def plot_learned_filters(filters):
    num_filters = len(filters)
    rows = math.ceil(num_filters / 4)
    cols = min(num_filters, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_filters:
            axes[i].imshow(filters[i], cmap="gray")
            axes[i].set_title(f"Filter {i+1}")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("Learned Filters (PCA)", fontsize=16)
    plt.savefig("3.2-filters.png")
    plt.show()

def plot_filter_responses(filtered_responses, original_image):
    num_responses = len(filtered_responses)
    total_plots = num_responses + 1

    rows, cols = 2, 5

    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i in range(1, len(axes)):
        if i <= num_responses:
            axes[i].imshow(filtered_responses[i - 1], cmap="gray")
            axes[i].set_title(f"Response {i}")
            axes[i].axis("off")
        else:
            axes[i].axis("off")

    plt.suptitle("Image Responses to Learned Filters", fontsize=16)
    plt.tight_layout()
    plt.savefig("3.2-responses.png")
    plt.show()

plot_learned_filters(filters)
plot_filter_responses(filtered_responses, gray_image)

"""### 3.3"""

features_njet = np.stack([response.flatten() for response in filter_responses.values()], axis=1)
pca_njet = PCA(n_components=3)
features_njet_pca = pca_njet.fit_transform(features_njet)

features_pca = np.stack([filtered_responses[i].flatten() for i in range(3)], axis=1)

kmeans_njet = KMeans(n_clusters=3, random_state=42).fit(features_njet_pca)
kmeans_pca = KMeans(n_clusters=3, random_state=42).fit(features_pca)

seg_njet = kmeans_njet.labels_.reshape(gray_image.shape)
seg_pca = kmeans_pca.labels_.reshape(gray_image.shape)

fig, ax = plt.subplots(1, 3, figsize=(16, 10))
ax[0].imshow(gray_image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(seg_njet, cmap="viridis")
ax[1].set_title("N-Jet + PCA")
ax[1].axis("off")

ax[2].imshow(seg_pca, cmap="viridis")
ax[2].set_title("PCA-Based Features")
ax[2].axis("off")

plt.suptitle("K-Means Segmentation Using Different Feature Representations", fontsize=16)
plt.tight_layout()
plt.savefig("3.3-segmentation.png")
plt.show()

