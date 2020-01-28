import numpy as np

from cv.transforms.base import Transform
from cv.transforms.filter import Convolve1d
from cv.transforms.kernels import gradient_kernel


def normalize(x):
    return (x-x.min())/(x.max()-x.min())

def non_max_supression(image, gradient_magnitude, gradient_direction, threshold=0):
    image_row, image_col = gradient_magnitude.shape

    all_pos = np.where(gradient_magnitude >= threshold)

    displacements = gradient_direction[all_pos]

    p_dir_x = np.where((-0.39 <= displacements) & (displacements <= 0.39), [all_pos[0], (all_pos[1] + 1) % image_col],
                       [all_pos[0], all_pos[1]])
    n_dir_x = np.where((-0.39 <= displacements) & (displacements <= 0.39), [all_pos[0], (all_pos[1] - 1) % image_col],
                       [all_pos[0], all_pos[1]])

    p_dir_xy = np.where((0.39 < displacements) & (displacements < 1.17),
                        [(p_dir_x[0] + 1) % image_row, (p_dir_x[1] + 1) % image_col], [p_dir_x[0], p_dir_x[1]])
    n_dir_xy = np.where((0.39 < displacements) & (displacements < 1.17),
                        [(n_dir_x[0] - 1) % image_row, (n_dir_x[1] - 1) % image_col], [n_dir_x[0], n_dir_x[1]])

    p_dir_y = np.where(1.17 <= displacements, [(p_dir_xy[0] + 1) % image_row, p_dir_xy[1]], [p_dir_xy[0], p_dir_xy[1]])
    n_dir_y = np.where(1.17 <= displacements, [(n_dir_xy[0] - 1) % image_row, n_dir_xy[1]], [n_dir_xy[0], n_dir_xy[1]])

    real_vals = gradient_magnitude[all_pos[0], all_pos[1]]
    positive_vals = gradient_magnitude[p_dir_y[0], p_dir_y[1]]
    negative_vals = gradient_magnitude[n_dir_y[0], n_dir_y[1]]

    real_vals[(real_vals < positive_vals) | (real_vals < negative_vals)] = 0

    #output = np.zeros(gradient_magnitude.shape)
    for i in range(len(all_pos[0])):
        image[all_pos[0][i], all_pos[1][i]] = real_vals[i]
    return image


def shift_helper(arr, neib_value, shift=0, axis=0):
    #Roll the 2D array along axis with certain unity
    array = np.roll(arr == neib_value, shift=shift, axis=axis)

    # Cancel the last/first slice shifted to the first/last slice
    if axis == 0:
        if shift >= 0:
            array[:1, :] = 0
        else:
            array[-1:, :] = 0
        return array
    elif axis == 1:
        if shift >= 0:
            array[:, :1] = 0
        else:
            array[:, -1:] = 0
        return array


def hysteresis(image):
    array = np.zeros(image.shape)
    array[np.where(((image == 100)
                     & (shift_helper(image, 255, shift=-1, axis=0)
                        | shift_helper(image, 255, shift=1, axis=0)
                        | shift_helper(image, 255, shift=-1, axis=1)
                        | shift_helper(image, 255, shift=1, axis=1)
                        )))] = 255
    array[np.where(image == 255)] = 255
    return array


def threshold(img, low_threshold_ratio=0.2, high_threshold_ratio=0.4):
    highThreshold = img.max() * high_threshold_ratio
    lowThreshold = img.max() * low_threshold_ratio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(100)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    res[zeros_i, zeros_j] = 0

    return res


class GradientMagnitude(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient_x_1 = gradient_kernel(operator=kwargs['operator'], direction='x', axis=0)
        gradient_x_2 = gradient_kernel(operator=kwargs['operator'], direction='x', axis=1)
        gradient_y_1 = gradient_kernel(operator=kwargs['operator'], direction='y', axis=0)
        gradient_y_2 = gradient_kernel(operator=kwargs['operator'], direction='y', axis=1)

        x = Convolve1d(kernel=gradient_x_1,axis=0).process(image)
        x = Convolve1d(kernel=gradient_x_2, axis=1).process(x)
        y = Convolve1d(kernel=gradient_y_1, axis=0).process(image)
        y = Convolve1d(kernel=gradient_y_2, axis=1).process(y)

        gradient = np.sqrt(x**2+y**2,casting='unsafe')
        return gradient


class GradientDirection(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient_x_1 = gradient_kernel(operator=kwargs['operator'], direction='x', axis=0)
        gradient_x_2 = gradient_kernel(operator=kwargs['operator'], direction='x', axis=1)
        gradient_y_1 = gradient_kernel(operator=kwargs['operator'], direction='y', axis=0)
        gradient_y_2 = gradient_kernel(operator=kwargs['operator'], direction='y', axis=1)

        x = Convolve1d(kernel=gradient_x_1, axis=0).process(image)
        x = Convolve1d(kernel=gradient_x_2, axis=1).process(x)
        y = Convolve1d(kernel=gradient_y_1, axis=0).process(image)
        y = Convolve1d(kernel=gradient_y_2, axis=1).process(y)

        gradient_direction = np.arctan2(y, x)

        return gradient_direction


class CannyEdge(Transform):
    default_args = {'size': 1, 'operator': 'sobel', 'sigma': 3}

    def apply(self, image, **kwargs):
        gradient_magnitude = GradientMagnitude(size=kwargs['size'],
                                               operator=kwargs['operator'],
                                               sigma=kwargs['sigma']).process(image)
        gradient_direction = GradientDirection(size=kwargs['size'],
                                               operator=kwargs['operator'],
                                               sigma=kwargs['sigma']).process(image)

        output = non_max_supression(image, gradient_magnitude, gradient_direction)
        output = threshold(output)
        output = hysteresis(output)
        return output
