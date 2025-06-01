# mandelbrot_app.py

import streamlit as st
from numba import jit, prange
import matplotlib.pyplot as plt
import numpy as np
import math


@jit(nopython=True, parallel=True, fastmath=True)
def mandelbrot_smooth_optimized(width, height, x_min, x_max, y_min, y_max, max_iter):
    result = np.zeros((height, width), dtype=np.float32)

    scale_x = (x_max - x_min) / width
    scale_y = (y_max - y_min) / height

    log2 = math.log(2.0)

    for y in prange(height):
        zy = y_min + y * scale_y
        for x in range(width):
            zx = x_min + x * scale_x
            zr, zi = 0.0, 0.0
            cr, ci = zx, zy
            n = 0

            while zr * zr + zi * zi <= 4.0 and n < max_iter:
                zr2 = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr2
                n += 1

            if n < max_iter:
                mag_sq = zr * zr + zi * zi
                log_zn = 0.5 * math.log(mag_sq)
                nu = math.log(log_zn / log2) / log2
                result[y, x] = n + 1 - nu
            else:
                result[y, x] = n
    return result

def equalize_histogram(data):
    """Apply histogram equalization to a 2D numpy array."""
    hist, bins = np.histogram(data.flatten(), bins=512, density=True)
    cdf = hist.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize to [0,1]

    # Use linear interpolation of cdf to find new pixel values
    data_flat = np.interp(data.flatten(), bins[:-1], cdf)
    return data_flat.reshape(data.shape)

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Streamlit interface
st.title("Mandelbrot-Explorer")
st.sidebar.header("Parameters")

res = st.sidebar.slider("resolutiomn", 100, 8000, 800, step=100)
max_iter = st.sidebar.slider("max. iterations", 10, 1000, 300, step=10)
#x_min, x_max = st.sidebar.slider("Select the x range", min_value = -2.0, max_value = 1.0, value = (-2.0, 1.0))
st.sidebar.text("center position")
c_x = st.sidebar.number_input("x-coordinate", format="%0.10f", value = -0.5)
c_y = st.sidebar.number_input("y-coordinate", format="%0.10f", value = 0.0)
zoom = st.sidebar.select_slider("Zoom factor", options = [1, 5, 25, 125, 625, 3125, 15625, 78125], value = 1.0)


hist_eq_on = st.sidebar.toggle("Histogram eualization", value=True)
norm_on = st.sidebar.toggle("Normalize", value=False)
#xmin = st.sidebar.slider("x-min", -2.5, -0.5, -2.0)
#xmax = st.sidebar.slider("x-max", -0.5, 1.5, 1.0)
#ymin = st.sidebar.slider("y-min", -1.5, 0.0, -1.0)
#ymax = st.sidebar.slider("y-max", 0.0, 1.5, 1.0)
#width = st.sidebar.slider("Width", 100, 1000, 500, step=100)
#height = st.sidebar.slider("Height", 100, 1000, 500, step=100)
#max_iter = st.sidebar.slider("Max Iterations", 50, 1000, 200, step=50)

# Parameters
#res = 800
width, height = res, res
#x_min, x_max = -2.0, 1.0
#y_min, y_max = -1.5, 1.5
#zoom =  78125
delta = 2 / zoom
center = c_x, c_y
x_min, y_min = center[0] - delta, center[1] - delta
x_max, y_max = center[0] + delta, center[1] + delta
#max_iter = 10000

# Compute
_img = mandelbrot_smooth_optimized(width, height, x_min, x_max, y_min, y_max, max_iter)
img = _img

# Equalize smooth values
if hist_eq_on:
    img = mandelbrot_eq = equalize_histogram(img)

if norm_on:
    img = normalize(img)

# Plotting
plt.figure()
plt.imshow(img, extent=(x_min, x_max, y_min, y_max), cmap='turbo')
plt.axis('off')
st.pyplot(plt)


