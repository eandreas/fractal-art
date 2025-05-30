import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
width, height = 800, 800
max_iter = 500

# Set up complex plane
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

# Create coordinate grid
x = np.linspace(x_min, x_max, width)
y = np.linspace(y_min, y_max, height)
X, Y = np.meshgrid(x, y)
C = X + 1j * Y

# Initialize the fractal
Z = np.zeros_like(C)
mask = np.full(C.shape, True, dtype=bool)

# Orbital trap settings
trap_point = 0 + 0j  # Point trap center
trap_radius = 0.01   # Threshold for trap proximity

def trap_distance(z):
    """Combine point and cross orbital traps"""
    # Point trap: distance to trap point
    dist_point = np.abs(z - trap_point)
    # Cross trap: min distance to real or imag axis (like a cross)
    dist_cross = np.minimum(np.abs(z.real), np.abs(z.imag))
    # Combine both (you can experiment with min, max, or weighted sum)
    return np.minimum(dist_point, dist_cross)

# Create image
image = np.zeros(C.shape)

# Iterate Mandelbrot and trap
for i in range(max_iter):
    Z[mask] = Z[mask]**2 + C[mask]

    # Calculate orbital trap distance
    trap = trap_distance(Z)
    # Accumulate min distance seen so far
    image = np.where(trap < image, trap, image)
    if i == 0:
        image = trap  # initialize in the first iteration

    # Escape condition
    mask &= (np.abs(Z) < 2)

# Normalize for visualization
image = np.log(1 + image)
image = image / np.max(image)

# Plot
plt.figure(figsize=(8, 8))
plt.imshow(image, extent=[x_min, x_max, y_min, y_max], cmap='inferno')
plt.title("Mandelbrot with Point and Cross Orbital Traps")
plt.axis('off')
plt.show()
