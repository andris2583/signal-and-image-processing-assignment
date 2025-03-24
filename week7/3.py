def hough_line(image):
    height, width = image.shape
    
    diag_len = int(np.sqrt(height**2 + width**2))
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
    thetas = np.deg2rad(np.arange(0, 180))
    
    accumulator = np.zeros((2 * diag_len, len(thetas)))
    
    for x in range(width):
      for y in range(height):
          if(image[x][y] != 0):
            for t_idx, theta in enumerate(thetas):
                rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
                accumulator[rho, t_idx] += 1
    return accumulator, rhos, thetas

def find_hough_peaks(accumulator, rhos, thetas, threshold=0.5):
    peaks = []
    max_acc = np.max(accumulator)
    threshold_value = threshold * max_acc
    
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold_value:
                peaks.append((rhos[rho_idx], thetas[theta_idx]))
                accumulator[rho_idx, theta_idx] = 0
    
    return peaks
image = io.imread('cross.png')
accumulator, rhos, thetas = custom_hough_line(image)
peaks = find_hough_peaks(accumulator,rhos,thetas,threshold=0.7)

image = io.imread("coins.png")
edges = canny(image, sigma=3)
radii_range = np.arange(15, 50, 5)

hough_res = hough_circle(edges, radii_range)

accums, cx, cy, radii = hough_circle_peaks(hough_res, radii_range, threshold=0.36)

fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(8, 8))
ax[0].imshow(edges,cmap='gray')
ax[0].set_title("Canny Edges")
ax[1].imshow(image, cmap='gray')

for center_x, center_y, radius in zip(cx, cy, radii):
  circy, circx = circle_perimeter(center_y, center_x, radius)
  ax[1].plot(circx, circy, 'r')

plt.title("Detected Coins")
plt.show()

original_image = io.imread("corrected.png")
image = color.rgb2gray(color.rgba2rgb(original_image))

edges = canny(image, sigma=4)

radii_range = np.arange(5, 15, 1)

hough_res = hough_circle(edges, radii_range)

_, cx, cy, radius = hough_circle_peaks(hough_res, radii_range, threshold=0.6)
print(len(cx))
cx, cy, radius = cx[0], cy[0], radius[0]
for x_d in range(-1,1):
    for y_d in range(-1,1):
      original_image[cy+x_d][cx+y_d] = (255,0,0,255)
circle1 = plt.Circle((cx, cy), radius, color='r', fill=False)
ax = plt.gca()
ax.cla() 
ax.add_patch(circle1)
plt.imshow(original_image, cmap='gray')