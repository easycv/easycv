import numpy as np

from cv.errors.transforms import InvalidMethodError, InvalidArgsError


def gray_scale(image, method):
    if method == 'averaging':
        return np.average(image, axis=2).astype(np.uint8)
    elif method == 'luma':
        return np.average(image, weights=[0.299, 0.587, 0.114], axis=2).astype(np.uint8)
    elif method == 'desaturation':
        return ((image.max(axis=2) + image.min(axis=2)) / 2).astype(np.uint8)
    elif method == 'decomposition_max':
        return image.max(axis=2).astype(np.uint8)
    elif method == 'decomposition_min':
        return image.min(axis=2).astype(np.uint8)
    else:
        raise InvalidMethodError(('averaging', 'luma', 'desaturation', 'decomposition_max', 'decomposition_min'))


def filter_channels(image, channels):
    channels = np.array(channels)
    if any(channels > 2) or any(channels < 0):
        raise InvalidArgsError(('channels',))
    image[:, :, channels] = 0
    return image


