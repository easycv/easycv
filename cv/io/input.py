import os

from urllib.error import URLError
from urllib.request import urlopen

import cv2
import numpy as np

from cv.errors.io import ImageDownloadError, InvalidPathError


def valid_image_array(image_array):
    source_is_grayscale = len(image_array.shape) == 2
    source_is_color = len(image_array.shape) == 3 and image_array.shape[2] == 3
    return source_is_grayscale or source_is_color


def valid_image_source(source):
    source_is_str = isinstance(source, str)
    source_is_array = isinstance(source, np.ndarray)
    return source_is_str or (source_is_array and valid_image_array)


def open_image(path):
    try:
        if os.path.isfile(path):
            img = cv2.imread(path)
        else:
            response = urlopen(path)
            img = np.asarray(bytearray(response.read()), dtype="uint8")
            if response.getcode() != 200:
                raise ImageDownloadError(f'Failed to Download file, error {response.getcode()}.')
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if not isinstance(img, np.ndarray):
                raise InvalidPathError('The given path is not an image.')
        return img

    except URLError:
        raise InvalidPathError('File path is invalid.') from None


def get_image_array(image_source):
    if isinstance(image_source, str):
        return open_image(image_source)
    else:
        return np.copy(image_source)
