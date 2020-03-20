import cv2
from PIL import Image
from matplotlib import pyplot as plt

from cv.utils import nearest_square_side


def prepare_image_to_output(img_arr, rgb=True):
    if img_arr.min() >= 0 and img_arr.max() <= 255:
        if img_arr.dtype.kind != "i":
            if img_arr.min() >= 0 and img_arr.max() <= 1:
                img_arr = img_arr * 255
            img_arr = img_arr.astype("uint8")
    else:
        img_arr = cv2.normalize(img_arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    if rgb and len(img_arr.shape) == 3 and img_arr.shape[-1] == 3:
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    return img_arr


def save(img_arr, filename, format=None):
    img_arr = prepare_image_to_output(img_arr)
    im = Image.fromarray(img_arr)
    im.save(filename, format)


def show(img_arr, name="Image", wait_time=500):
    img_arr = prepare_image_to_output(img_arr, rgb=False)
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, img_arr.shape[0], img_arr.shape[1])
    cv2.imshow(name, img_arr)

    while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
        key_code = cv2.waitKey(wait_time)
        if (key_code & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            break


def show_grid(images, titles=(), size="auto"):
    if size == "auto":
        side = nearest_square_side(len(images))
        size = (side, side)
    for i in range(len(images)):
        img = prepare_image_to_output(images[i].array)
        plt.subplot(size[0], size[1], i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        if len(titles) == len(images):
            plt.title(titles[i])
    plt.show()
