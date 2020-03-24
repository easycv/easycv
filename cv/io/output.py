import cv2
from PIL import Image
from matplotlib import pyplot as plt

from cv.utils import nearest_square_side


def prepare_image_to_output(img_arr, rgb=True):
    """
    Formats the image array to be showed turning its values to be between 0 and 255 and puts image
    in BGR abstraction

    :param img_arr: image array
    :type img_arr: :class:`~numpy:numpy.ndarray`
    :param rgb: If image is in RGB abstraction
    :type rgb: :class:`bool`, optional
    :return: image array
    :rtype: :class:`~numpy:numpy.ndarray`
    """
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
    """
    Saves an image to a file

    :param img_arr: Image as an array
    :type img_arr: :class:`~numpy:numpy.ndarray`
    :param filename: Rgb flag
    :type filename: :class:`str`
    :param format: File Format, defaults to None
    :type format: :class:`str`, optional
    """
    img_arr = prepare_image_to_output(img_arr)
    im = Image.fromarray(img_arr)
    im.save(filename, format)


def show(img_arr, name="Image", wait_time=500):
    """
    Creates a cv2 window and displays the given image arrays

    :param img_arr: Image as an array
    :type img_arr: :class:`~numpy:numpy.ndarray`
    :param name: Window name, defaults to "Image"
    :type name: :class:`str`, optional
    :param wait_time: Time between cicles of waiting for end key, defaults to 500
    :type wait_time: :class:`int`, optional
    """
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
    """
    Display images in a grid

    :param images: Array of Images
    :type images: :class:`np.array`
    :param titles: Image titles
    :type titles: :class:`tuple`
    :param size: Size of a image, defaults to "auto"
    :type size: :class:`str`, optional
    """
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
