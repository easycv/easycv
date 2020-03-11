
import cv2
from PIL import Image


def prepare_image_to_output(img_arr):

    if img_arr.min() >= 0 and img_arr.max() <= 255:
        if img_arr.dtype.kind != 'i':
            img_arr = img_arr.astype('uint8')
    else:
        img_arr = cv2.normalize(img_arr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    if len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    return img_arr


def save(img_arr, filename, format=None):
    im = Image.fromarray(img_arr)
    im.save(filename, format)

