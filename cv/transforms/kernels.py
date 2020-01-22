import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


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
