from matplotlib.widgets import RectangleSelector, EllipseSelector
import matplotlib.pyplot as plt

from easycv.io.output import prepare_image_to_output


def toggle_selector(event):  # Bindings to close the matplotlib window
    if event.key in ['Q', 'q'] and toggle_selector.S.active:
        toggle_selector.S.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.S.active:
        toggle_selector.S.set_active(True)


def rectangle_selector(image):
    """
    Allows the user to select a rectangle region in an image

    :param image: Image to display
    :type image: :class:`Image`
    :return: Array with the points of the upper left and bottom right corners
    :rtype: :class:`list`
    """
    def line_select_callback(e1, e2):  # Callback to RectangleSelector
        pass
    fig, current_ax = plt.subplots()
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    current_ax.imshow(prepare_image_to_output(image.array))
    toggle_selector.S = RectangleSelector(current_ax, line_select_callback, useblit=True,
                                            button=[1, 3], minspanx=5, minspany=5,
                                            spancoords='pixels', interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show(block=True)
    x, y = toggle_selector.S.to_draw.get_xy()
    x = int(round(x))
    y = int(round(y))
    width = round(toggle_selector.S.to_draw.get_width())
    height = round(toggle_selector.S.to_draw.get_height())
    if width != 0 or height != 1:
        return [(x, y), (x+width, y+height)]


def point_selector(image, n_points):
    """
    Allows the user to select a number of points in an image

    :param image: Image to display
    :type image: :class:`Image`
    :param n_points: Number of points to select
    :type n_points: :class:`int`
    :return: Array with points selected
    :rtype: :class:`list`
    """
    def onclick(event):
        res.append((round(event.xdata), round(event.ydata)))
        plt.plot(event.xdata, event.ydata, marker='o', color='cyan', markersize=4)
        fig.canvas.draw()
        if len(res) == n_points:
            plt.close(fig)
    res = []
    fig, current_ax = plt.subplots()
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    current_ax.imshow(prepare_image_to_output(image.array))
    plt.gcf().canvas.set_window_title('Point Selector')
    plt.connect('button_press_event', onclick)
    plt.show(block=True)
    return res


def ellipse_selector(image):
    """
    Allows the user to select an ellipse region in an image

    :param image: Image to display
    :type image: :class:`Image`
    :return: Dictionary with the center, width and height of the ellipse
    :rtype: :class:`list`
    """
    def onselect(e1, e2):
        pass
    fig, current_ax = plt.subplots()
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False,
                    labelbottom=False, labelleft=False)
    toggle_selector.S = EllipseSelector(current_ax, onselect, drawtype='box', interactive=True,
                                        useblit=True)
    current_ax.imshow(prepare_image_to_output(image.array))
    fig.canvas.mpl_connect('key_press_event', toggle_selector)
    plt.show(block=True)
    width = round(toggle_selector.S.to_draw.width)
    height = round(toggle_selector.S.to_draw.height)
    center = [round(x) for x in toggle_selector.S.to_draw.get_center()]
    return {"center": center, "width": width, "height": height}

