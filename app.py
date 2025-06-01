# mandelbrot_app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to compute Mandelbrot set
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    Z = np.zeros_like(C)
    div_time = np.zeros(C.shape, dtype=int)

    for i in range(max_iter):
        Z = Z**2 + C
        diverge = np.abs(Z) > 2
        div_now = diverge & (div_time == 0)
        div_time[div_now] = i
        Z[diverge] = 2

    return div_time.T  # Transpose for correct orientation

# Streamlit interface
st.title("Mandelbroti-Explorer")
st.sidebar.header("Parameters")

xmin = st.sidebar.slider("x-min", -2.5, -0.5, -2.0)
xmax = st.sidebar.slider("x-max", -0.5, 1.5, 1.0)
ymin = st.sidebar.slider("y-min", -1.5, 0.0, -1.0)
ymax = st.sidebar.slider("y-max", 0.0, 1.5, 1.0)
width = st.sidebar.slider("Width", 100, 1000, 500, step=100)
height = st.sidebar.slider("Height", 100, 1000, 500, step=100)
max_iter = st.sidebar.slider("Max Iterations", 50, 1000, 200, step=50)

# Compute the Mandelbrot set
mandelbrot_img = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Plotting
#plt.figure(figsize=(10, 10))
plt.figure()
plt.imshow(mandelbrot_img, extent=(xmin, xmax, ymin, ymax), cmap='turbo')
plt.axis('off')
st.pyplot(plt)


