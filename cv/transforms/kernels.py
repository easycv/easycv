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


def gradient_kernel(operator='sobel'):
    if operator == 'sobel':
        kernel_x = (Convolve(kernel=np.array([[-1, 0, 1]])).process(np.array([[1, 2, 1]]).T))
        kernel_y = Convolve(kernel=np.array([[-1, 0, 1]]).T).process(np.array([[1, 2, 1]]))
    elif operator == 'prewitt':
        kernel_x = np.ones((3, 3))
        kernel_y = np.ones((3, 3))
        kernel_x[:, 0] *= -1
        kernel_x[:, 2] = 0
        kernel_y[2, :] *= -1
        kernel_y[1, :] = 0
    elif operator == 'roberts':
        kernel_x = np.zeros((2, 2))
        kernel_y = np.zeros((2, 2))
        kernel_x[0, 1] = 1
        kernel_x[1, 0] = -1
        kernel_y[0, 0] = 1
        kernel_y[1, 1] = -1
    else:
        raise
    kernel_sum = np.abs(np.sum(kernel_x[:, :1])*2)
    kernel_x = np.true_divide(kernel_x, kernel_sum)
    kernel_sum = np.abs(np.sum(kernel_y[:1, :])*2)
    kernel_y = np.true_divide(kernel_y, kernel_sum)
    return kernel_x, kernel_y


def smooth_gradient_kernel(size=3, operator='sobel', sigma=3):
    gauss = gaussian_kernel(size=size, sig=sigma)
    gradient = gradient_kernel(operator=operator)
    kernel_x = Convolve(kernel=gradient[0]).process(gauss)
    kernel_y = Convolve(kernel=gradient[1]).process(gauss)
    return kernel_x, kernel_y
