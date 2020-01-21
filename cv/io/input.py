import requests
import numpy as np
from PIL import Image

from cv.errors.io import ImageDownloadError, InvalidPathError

def openImage(img_dir):
    try:
        img = Image.open(img_dir)
    except IOError as ioe:
        raise InvalidPathError('Invalid Path')
    return img


def downloadImage(url, path):
    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        raise ImageDownloadError('Invalid Download')
    open(path, 'wb').write(response.content)
    return openImage(path)
