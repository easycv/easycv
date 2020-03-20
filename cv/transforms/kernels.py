import numpy as np
import matplotlib.pyplot as plt

from cv.transforms.filter import Convolve1d


def view_kernel(kernel, size=(5, 5), legend=False):
    plt.figure(figsize=size)
    plt.imshow(kernel)
    if legend:
        plt.colorbar(orientation="vertical")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def uniform_1d(radius=3):
    return np.ones(radius * 2 + 1) / radius * 2 + 1


def gaussian_1d(sigma, radius):
    x = np.arange(-radius, radius + 1)
    variance = sigma * sigma
    gaussian = np.exp(-0.5 * (x ** 2) / variance)
    return gaussian / gaussian.sum()


def gradient_kernel(operator="sobel", direction="x", axis=0):
    if operator == "sobel":
        if direction == "x":
            if not axis:
                kernel = [1, 2, 1]
            else:
                kernel = [1, 0, -1]
        else:
            if not axis:
                kernel = [-1, 0, 1]
            else:
                kernel = [1, 2, 1]
    return kernel


def smooth_gradient_kernel(size=3, operator="sobel", direction="x", sigma=3):
    gauss = gaussian_1d(radius=size, sigma=sigma)
    gradient_x = gradient_kernel(operator=operator, direction=direction, axis=0)
    gradient_y = gradient_kernel(operator=operator, direction=direction, axis=1)
    kernel_x = Convolve1d(kernel=gradient_x).process(gauss)
    kernel_y = Convolve1d(kernel=gradient_y).process(gauss)
    return kernel_x, kernel_y
