import matplotlib.pyplot as plt
from skimage import io, feature
'''
### 3.1
'''
gray_image = io.imread('/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 5/Assignment/hand.tiff', as_gray=True)

sigma_values = [0.1, 2, 3, 7]
low_threshold_values = [0, 50, 100]
high_threshold_values = [0, 50, 100]

valid_threshold_pairs = [(low, high) for low in low_threshold_values for high in high_threshold_values if low <= high]

def plot_canny_edges():
    fig, axes = plt.subplots(len(sigma_values), len(valid_threshold_pairs) + 1, figsize=(15, 12))

    for row, sigma in enumerate(sigma_values):
        axes[row, 0].imshow(gray_image, cmap='gray')
        axes[row, 0].set_title(f"Original Image (Sigma={sigma})")
        axes[row, 0].axis('off')

        for col, (low_thresh, high_thresh) in enumerate(valid_threshold_pairs):
            edges = feature.canny(gray_image, sigma=sigma, low_threshold=low_thresh, high_threshold=high_thresh)

            axes[row, col + 1].imshow(edges, cmap='gray')
            axes[row, col + 1].set_title(f"Low={low_thresh}, High={high_thresh}")
            axes[row, col + 1].axis('off')

    plt.suptitle("Canny Edge Detection for Different Sigma and Thresholds", fontsize=16)
    plt.tight_layout()
    plt.show()

plot_canny_edges()

"""
### 3.2
"""

image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 5/Assignment/modelhouses.png"
gray_image = io.imread(image_path, as_gray=True)

sigma_values_k = [1, 5, 10, 15, 20]
k_values = [0.001, 0.01, 0.1, 0.6, 1]

def plot_k_method():
    fig, axes = plt.subplots(len(sigma_values_k), len(k_values) + 1, figsize=(15, 10))

    for row, sigma in enumerate(sigma_values_k):
        axes[row, 0].imshow(gray_image, cmap='gray')
        axes[row, 0].set_title(f"Original (Sigma={sigma})")
        axes[row, 0].axis('off')

        for col, k in enumerate(k_values):
            corner_response = feature.corner_harris(gray_image, method='k', sigma=sigma, k=k)

            axes[row, col + 1].imshow(corner_response, cmap='hot')
            axes[row, col + 1].set_title(f"K={k}, Sigma={sigma}")
            axes[row, col + 1].axis('off')

    plt.suptitle("Harris Corner Detection - K Method", fontsize=16)
    plt.tight_layout()
    plt.show()

sigma_values_eps = [1, 2, 3, 5, 8]
eps_values = [0.001, 0.01, 0.1, 0.6, 1]

def plot_eps_method():
    fig, axes = plt.subplots(len(sigma_values_eps), len(eps_values) + 1, figsize=(15, 10))

    for row, sigma in enumerate(sigma_values_eps):
        axes[row, 0].imshow(gray_image, cmap='gray')
        axes[row, 0].set_title(f"Original (Sigma={sigma})")
        axes[row, 0].axis('off')

        for col, eps in enumerate(eps_values):
            corner_response = feature.corner_harris(gray_image, method='eps', sigma=sigma, eps=eps)
            axes[row, col + 1].imshow(corner_response, cmap='hot')
            axes[row, col + 1].set_title(f"Eps={eps}, Sigma={sigma}")
            axes[row, col + 1].axis('off')

    plt.suptitle("Harris Corner Detection - Eps Method (Varying Sigma)", fontsize=16)
    plt.tight_layout()
    plt.show()

plot_k_method()

plot_eps_method()

"""### 3.3"""

image_path = "/content/drive/Othercomputers/Laptop/Documents/KU/SIP/Week 5/Assignment/modelhouses.png"
gray_image = io.imread(image_path)

def detect_harris_corners(image, sigma=1.5, k=0.05, num_peaks=250):
    corner_response = feature.corner_harris(image, method='k', sigma=sigma, k=k)

    corners = feature.corner_peaks(corner_response, num_peaks=num_peaks, threshold_rel=0.01)

    print(f"Detected {len(corners)} corners")

    fig, ax = plt.subplots(1, 2, figsize=(22, 10))

    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(image, cmap='gray')
    ax[1].scatter(corners[:, 1], corners[:, 0], s=40, c='red', edgecolors='black', linewidth=1.2, label="Detected Corners")
    ax[1].set_title(f"Harris Corners (Top {num_peaks} Strongest)")
    ax[1].axis('off')
    ax[1].legend()

detect_harris_corners(gray_image)
