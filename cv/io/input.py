import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from requests.exceptions import InvalidSchema, MissingSchema

from cv.errors.io import ImageDownloadError, InvalidPathError


def open_image(path):
    try:
        if os.path.isfile(path):
            img = Image.open(path)
        else:
            response = requests.get(path, allow_redirects=True)
            if response.status_code != 200:
                raise ImageDownloadError(f'Failed to Download file, error {response.status_code}.')
            img = Image.open(BytesIO(response.content))
        return np.array(img)

    except (ConnectionError, InvalidSchema, MissingSchema):
        raise InvalidPathError('File path is invalid.') from None

    except UnidentifiedImageError:
        raise InvalidPathError('The given path is not an image.') from None
