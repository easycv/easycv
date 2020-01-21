import numpy as np
from PIL import Image

Image._show()
def open(img_dir):
    return np.array(Image.open(img_dir))
