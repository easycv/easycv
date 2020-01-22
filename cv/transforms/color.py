import numpy as np


def gray_scale(image, method):
    if method == 'averaging':
        return np.average(image, axis=2).astype(np.uint8)
    elif method == 'luma':
        return np.average(image, weights=[0.3, 0.59, 0.11], axis=2).astype(np.uint8)
    elif method == 'desaturation':
        return ((image.max(axis=2) + image.min(axis=2)) / 2).astype(np.uint8)
    elif method == 'decomposition_max':
        return image.max(axis=2).astype(np.uint8)
    elif method == 'decomposition_min':
        return image.min(axis=2).astype(np.uint8)
    else:
        print(f"Method {method} not suported")


def filter_channels(image, channels):
    for channel in channels:
        image[:, :, channel] = 0
    return image

