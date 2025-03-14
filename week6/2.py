import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, transform

image = io.imread("textlabel_gray_small.png", as_gray=True)

def harris_corner(image, sigma=1.5, alpha=0.04):
  
    Ix = filters.sobel_v(image)
    Iy = filters.sobel_h(image)

  
    Lxx = Ix**2
    Lyy = Iy**2
    Lxy = Ix * Iy

  
    Lxx = filters.gaussian(Lxx, sigma=sigma)
    Lyy = filters.gaussian(Lyy, sigma=sigma)
    Lxy = filters.gaussian(Lxy, sigma=sigma)

  
    det_A = (Lxx * Lyy) - (Lxy ** 2)
    trace_A = Lxx + Lyy
    R = det_A - alpha * (trace_A ** 2)

    return R

R = harris_corner(image)

threshold = 0.01 * R.max()
corners = np.argwhere(R > threshold)

def find_edge_corners(corners):
    sorted_corners = sorted(corners, key=lambda x: x[1]) 
    left = sorted_corners[:int(len(corners)*0.05)]  
    right = sorted_corners[-int(len(corners)*0.05):]  

    top_left = min(left, key=lambda x: x[0])
    bottom_left = max(left, key=lambda x: x[0])
    top_right = min(right, key=lambda x: x[0])
    bottom_right = max(right, key=lambda x: x[0])
    return np.array([top_left, bottom_left, top_right, bottom_right])

edge_corners = find_edge_corners(corners)

def compute_rotation_angle(corners):
    top_left, bottom_left, top_right, bottom_right = corners
    angle_left = np.arctan2(bottom_left[0] - top_left[0], bottom_left[1] - top_left[1]) * 180 / np.pi
    angle_right = np.arctan2(bottom_right[0] - top_right[0], bottom_right[1] - top_right[1]) * 180 / np.pi
    angle_top = np.arctan2(top_left[0] - top_right[0], top_left[1] - top_right[1]) * 180 / np.pi
    angle_bottom = np.arctan2(bottom_left[0] - bottom_right[0], bottom_right[1] - bottom_left[1]) * 180 / np.pi

    angle = (angle_left + angle_right + angle_top + angle_bottom) / 4 
    return angle

rotated_image = transform.rotate(image, -compute_rotation_angle(edge_corners), resize=True)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap="gray")
axes[0].scatter(corners[:, 1], corners[:, 0], c='red', s=10)
axes[0].set_title("All Detected Corners")
axes[1].imshow(image, cmap="gray")
axes[1].scatter(edge_corners[:, 1], edge_corners[:, 0], c='red', s=10)
axes[1].set_title("Detected Edge Corners")
axes[2].imshow(rotated_image, cmap="gray")
axes[2].set_title(f"Rotated Image ({-compute_rotation_angle(edge_corners):.2f}°)")
axes[2].axis("off")

plt.tight_layout()
plt.show()