import os

import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector

import easycv.image
from easycv.transforms.base import Transform
from easycv.errors import InvalidSelectionError
from easycv.validators import Number, List, Type, Image
from easycv.io.output import prepare_image_to_output


class Select(Transform):
    """
    Select is a transform that allows the user to select a shape or a mask in an image. Currently \
    supported shapes:

    \t**∙ rectangle** - Rectangle Shape\n
    \t**∙ point** - Point\n
    \t**∙ ellipse** - Ellipse Shape\n
    \t**∙ mask** - Mask\n

    :param n: Number of points to select
    :type n: :class:`int`
    :param brush: Brush size
    :type brush: :class:`int`
    :param color: Brush color
    :type color: :class:`List`
    """

    methods = {
        "rectangle": {"arguments": [], "outputs": ["rectangle"]},
        "point": {"arguments": ["n"], "outputs": ["points"]},
        "ellipse": {"arguments": [], "outputs": ["ellipse"]},
        "mask": {"arguments": ["brush", "color"], "outputs": ["mask"]},
    }
    default_method = "rectangle"

    arguments = {
        "n": Number(only_integer=True, min_value=0, default=2),
        "brush": Number(only_integer=True, min_value=0, default=20),
        "color": List(
            Number(only_integer=True, min_value=0, max_value=255),
            length=3,
            default=(0, 255, 0),
        ),
    }

    outputs = {
        # rectangle
        "rectangle": List(
            List(Number(min_value=0, only_integer=True), length=2), length=2
        ),
        # ellipse
        "ellipse": List(
            List(Number(only_integer=True, min_value=0), length=2),
            Number(min_value=0, only_integer=True),
            Number(min_value=0, only_integer=True),
        ),
        # point
        "points": List(List(Number(min_value=0, only_integer=True), length=2)),
        # mask
        "mask": Image(),
    }

    def process(self, image, **kwargs):
        if "DISPLAY" not in os.environ:
            raise Exception("Can't run selectors without a display!")

        if kwargs["method"] == "mask":
            mask = np.zeros(image.shape, np.uint8)

            global drawing
            drawing = False

            def paint_draw(event, x, y, flags, param):
                global ix, iy, drawing

                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    cv2.line(mask, (ix, iy), (x, y), kwargs["color"], kwargs["brush"])

                ix, iy = x, y

                return x, y

            cv2.namedWindow("Select Mask", cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow("Select Mask", image.shape[0], image.shape[1])
            cv2.setMouseCallback("Select Mask", paint_draw)

            while cv2.getWindowProperty("Select Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.imshow("Select Mask", cv2.addWeighted(image, 0.8, mask, 0.2, 0))
                key_code = cv2.waitKey(1)

                if (key_code & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    break
                elif (key_code & 0xFF) == ord("+"):
                    kwargs["brush"] += 1
                elif (key_code & 0xFF) == ord("-") and kwargs["brush"] > 1:
                    kwargs["brush"] -= 1

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask[mask != 0] = 255

            return {"mask": easycv.image.Image(mask)}

        mpl.use("Qt5Agg")

        fig, current_ax = plt.subplots()
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        def empty_callback(e1, e2):
            pass

        def selector(event):
            if event.key in ["Q", "q"]:
                plt.close(fig)

        res = []
        current_ax.imshow(prepare_image_to_output(image))
        plt.gcf().canvas.set_window_title("Selector")

        if kwargs["method"] == "rectangle":
            selector.S = RectangleSelector(
                current_ax,
                empty_callback,
                useblit=True,
                button=[1, 3],
                minspanx=5,
                minspany=5,
                spancoords="pixels",
                interactive=True,
            )
        elif kwargs["method"] == "ellipse":
            selector.S = EllipseSelector(
                current_ax,
                empty_callback,
                drawtype="box",
                interactive=True,
                useblit=True,
            )
        else:

            def onclick(event):
                if event.xdata is not None and event.ydata is not None:
                    res.append((int(event.xdata), int(event.ydata)))
                    plt.plot(
                        event.xdata, event.ydata, marker="o", color="cyan", markersize=4
                    )
                    fig.canvas.draw()
                    if len(res) == kwargs["n"]:
                        plt.close(fig)

            plt.connect("button_press_event", onclick)

        plt.connect("key_press_event", selector)
        plt.show(block=True)

        if kwargs["method"] == "rectangle":
            x, y = selector.S.to_draw.get_xy()
            x = int(round(x))
            y = int(round(y))
            width = int(round(selector.S.to_draw.get_width()))
            height = int(round(selector.S.to_draw.get_height()))

            if width == 0 or height == 0:
                raise InvalidSelectionError("Must select a rectangle.")

            return {"rectangle": [(x, y), (x + width, y + height)]}

        elif kwargs["method"] == "ellipse":
            width = int(round(selector.S.to_draw.width))
            height = int(round(selector.S.to_draw.height))
            center = [int(round(x)) for x in selector.S.to_draw.get_center()]
            if width == 0 or height == 0:
                raise InvalidSelectionError("Must select an ellipse.")
            return {"ellipse": [tuple(center), int(width / 2), int(height / 2)]}
        else:
            if len(res) != kwargs["n"]:
                raise InvalidSelectionError(
                    "Must select {} points.".format(kwargs["n"])
                )
            return {"points": res}


class Mask(Transform):
    """
    Mask applies a mask to an image.

    :param mask: Mask to apply
    :type brush: :class:`Image`
    :param inverse: Inverts mask
    :type inverse: :class:`bool`
    :param fill_color: Color to fill
    :type fill_color: :class:`List`
    """

    arguments = {
        "mask": Image(),
        "inverse": Type(bool, default=False),
        "fill_color": List(
            Number(only_integer=True, min_value=0, max_value=255),
            length=3,
            default=(0, 0, 0),
        ),
    }

    def process(self, image, **kwargs):

        if kwargs["inverse"]:
            mask = cv2.bitwise_not(kwargs["mask"].array)
        else:
            mask = kwargs["mask"].array

        image = cv2.bitwise_and(image, image, mask=mask)
        image[mask == 0] = kwargs["fill_color"]

        return image


class Inpaint(Transform):
    """
    Inpaint applies an inpainting technique to an image.

    :param radius: Inpainting radius
    :type radius: :class:`int`
    :param mask: Mask to apply inpaint
    :type mask: :class:`Image`
    """

    methods = {
        "telea": {"arguments": ["radius", "mask"]},
        "ns": {"arguments": ["radius", "mask"]},
    }
    default_method = "telea"

    arguments = {
        "radius": Number(only_integer=True, min_value=0, default=3),
        "mask": Image(),
    }

    def process(self, image, **kwargs):
        flag = cv2.INPAINT_TELEA if kwargs["method"] == "telea" else cv2.INPAINT_NS

        return cv2.inpaint(image, kwargs["mask"].array, kwargs["radius"], flags=flag)
