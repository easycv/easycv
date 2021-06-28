import os

from urllib.error import URLError
from urllib.request import urlopen
from json import loads

import cv2
import numpy as np

from easycv.errors.io import ImageDownloadError, InvalidPathError


def valid_image_array(image_array):
    """
    Returns `True` if and image array is valid.
    An image array is valid if it is in grayscale (having 2 dimensions) or in color
    (having 3 dimensions)

    :param image_array: Image as an array
    :type image_array: :class:`~numpy:numpy.ndarray`
    :return: Returns `True` if an image array is valid, otherwise `False`
    :rtype: :class:`bool`
    """
    source_is_grayscale = len(image_array.shape) == 2
    source_is_color = len(image_array.shape) == 3 and image_array.shape[2] == 3
    return source_is_grayscale or source_is_color


def valid_image_source(source):
    """
    Returns `True` if a source is valid
    A source is valid if it is a string or a :class:`~numpy:numpy.ndarray`

    :param source: Source of an image
    :type source: :class:`str`/:class:`~numpy:numpy.ndarray`
    :return: Returns `True` if a source is valid, otherwise `False`
    :rtype: :class:`bool`
    """
    source_is_str = isinstance(source, str)
    source_is_array = isinstance(source, np.ndarray)
    return source_is_str or (source_is_array and valid_image_array(source))


def open_image(path):
    """
    Opens/Downloads an image and reads it into an array

    :param path: Path/Link to an image
    :type path: :class:`str`
    :return: Image as an array
    :rtype: :class:`~numpy:numpy.ndarray`
    """
    try:
        if os.path.isfile(path):
            img = cv2.imread(path)
        else:
            response = urlopen(path)
            if response.getcode() != 200:
                raise ImageDownloadError(
                    "Failed to Download file, error {}.".format(response.getcode())
                )
            img = np.asarray(bytearray(response.read()), dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if not isinstance(img, np.ndarray):
                raise InvalidPathError("The given path is not an image.")
        return img

    except URLError:
        raise InvalidPathError("File path is invalid.") from None


def random_dog_image():
    """
    Makes a request to `DogApi <https://dog.ceo/dog-api/>`_ for a random image and
    extracts the link from the response.

    :return: Link to a random dog image
    :rtype: :class:`str`
    """
    response = urlopen("https://dog.ceo/api/breeds/image/random")
    if response.getcode() != 200:
        raise ImageDownloadError(
            "Failed to Download file, error {}.".format(response.getcode())
        )
    buf = response.read()
    result = loads(buf.decode("utf-8"))
    return result["message"]


def get_image_array(image_source):
    """
    :param image_source: Path/Link to an image or an array of an image
    :type image_source: :class:`~numpy:numpy.ndarray`/:class:`str`
    :return: image as an array
    :rtype: :class:`~numpy:numpy.ndarray`
    """
    if isinstance(image_source, str):
        return open_image(image_source)
    else:
        return np.copy(image_source)


def open_folder(list_source, recursive=False):
    """
    Searches in the folder path given for the images present

    :param list_source: Path to a folder of images
    :type list_source: :class:`str`
    :param recursive: Flag to allow search in the all directories of the folder
    :type recursive: :class:`bool`
    :return: list of images
    :rtype: :class:`list`
    """
    if not recursive:
        paths = [os.path.join(list_source, fn) for fn in next(os.walk(list_source))[2]]
    else:
        paths = []
        for root, _, files in os.walk(list_source):
            for filename in files:
                paths.append(os.path.join(root, filename))
    images = []
    for file in paths:
        tmp = get_image_array(file)
        if tmp is not None:
            images.append(tmp)
    return images


def get_image_list(list_source, recursive=False):
    """
    Gets all the images from a folder

    :param list_source: Path to a folder of images
    :type list_source: :class:`str`
    :param recursive: Flag to allow search in the all directories of the folder
    :type recursive: :class:`bool`
    :return: list of images
    :rtype: :class:`list`
    """
    if isinstance(list_source, str):
        return open_folder(list_source, recursive=recursive)
    else:
        return np.copy(list_source)
