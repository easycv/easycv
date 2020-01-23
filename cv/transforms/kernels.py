import numpy as np
import matplotlib.pyplot as plt

from cv.transforms.filter import Convolve


def view_kernel(kernel, size=(5, 5), legend=False):
    plt.figure(figsize=size)
    plt.imshow(kernel)
    if legend:
        plt.colorbar(orientation='vertical')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def uniform_kernel(size=21):
    """Returns a 2D uniform kernel."""
    return np.ones((size, size)) / size ** 2


def gaussian_kernel(size=21, sig=3):
    """Returns a 2D Gaussian kernel."""
    ax = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


def gradient_kernel(operator='sobel', axis='x'):
    if operator not in ['sobel', 'prewitt', 'roberts']:
        raise
    elif axis not in ['x','y']:
        raise
    if operator == 'sobel':
        if axis == 'x':
            kernel = (Convolve(kernel=np.array([[-1, 0, 1]])).apply(np.array([[1, 2, 1]]).T))
        elif axis == 'y':
            kernel = Convolve(kernel=np.array([[-1, 0, 1]]).T).apply(np.array([[1, 2, 1]]))
    elif operator == 'prewitt':
        kernel = np.ones((3, 3))
        if axis == 'x':
            kernel[:, 0] *= -1
            kernel[:, 2] = 0
        elif axis == 'y':
            kernel[2, :] *= -1
            kernel[1, :] = 0
    elif operator == 'roberts':
        kernel = np.zeros((2, 2))
        if axis == 'x':
            kernel[0, 1] = 1
            kernel[1, 0] = -1
        elif axis == 'y':
            kernel[0, 0] = 1
            kernel[1, 1] = -1
    if axis == 'x':
        kernel_sum = np.abs(np.sum(kernel[:, :1])*2)
    elif axis == 'y':
        kernel_sum = np.abs(np.sum(kernel[:1, :])*2)
    return kernel/kernel_sum


def smooth_gradient_kernel(size=3, operator='sobel', axis='x', sigma=3):
    gauss = gaussian_kernel(size=size, sig=sigma)
    kernel = Convolve(kernel=gradient_kernel(operator, axis)).apply(gauss)
    return kernel
