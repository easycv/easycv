from PIL import Image


def save(img_arr, filename, format=None):
    im = Image.fromarray(img_arr)
    im.save(filename, format)

