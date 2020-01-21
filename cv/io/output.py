from PIL import Image


def save(img_arr, filename):
    im = Image.fromarray(img_arr)
    im.save(filename)

