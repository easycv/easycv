import os
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from requests.exceptions import MissingSchema

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

    except ConnectionError:
        raise InvalidPathError('Path to file is invalid.') from None

    except MissingSchema:
        raise InvalidPathError('Path to file is invalid.') from None

