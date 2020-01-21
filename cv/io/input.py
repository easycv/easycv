from PIL import Image
import numpy as np


def open(img_dir):
    return np.array(Image.open(img_dir))
