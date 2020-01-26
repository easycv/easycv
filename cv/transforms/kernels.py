import numpy as np
import matplotlib.pyplot as plt

from cv.transforms.filter import Convolve1d


def view_kernel(kernel, size=(5, 5), legend=False):
    plt.figure(figsize=size)
    plt.imshow(kernel)
    if legend:
        plt.colorbar(orientation='vertical')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def uniform_1d(radius=3):
    return np.ones(radius*2 + 1) / radius*2 + 1


def gaussian_1d(sigma, radius):
    x = np.arange(-radius, radius+1)
    variance = sigma * sigma
    gaussian = np.exp(-0.5 * (x ** 2) / variance)
    return gaussian / gaussian.sum()


def gradient_kernel(operator='sobel'):
    if operator == 'sobel':
        kernel = ([1, 2, 1], [-1, 0, 1])
    elif operator == 'prewitt':
        kernel = ([1, 1, 1], [1, 0, -1])
    #elif operator == 'roberts':
    #    kernel_x = np.zeros((2, 2))
    #    kernel_y = np.zeros((2, 2))
    #    kernel_x[0, 1] = 1
    #    kernel_x[1, 0] = -1
    #    kernel_y[0, 0] = 1
    #    kernel_y[1, 1] = -1# todo change to x,y vectors to convolve/correlate
    else:
        raise
    return kernel


def smooth_gradient_kernel(size=3, operator='sobel', sigma=3):
    gauss = gaussian_1d(radius=size, sigma=sigma)
    gradient = gradient_kernel(operator=operator)
    kernel_x = Convolve1d(kernel=gradient[0]).process(gauss)
    kernel_y = Convolve1d(kernel=gradient[1]).process(gauss)
    return kernel_x, kernel_y
