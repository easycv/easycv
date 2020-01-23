import numpy as np

from cv.errors.transforms import InvalidArgumentError, InvalidMethodError
from cv.transforms.base import Transform


class CropImage(Transform):
    arguments = {'box': []}

    def apply(self, image):
        box = self.arguments['box']
        if type(box) == list and len(box) == 4 and all([type(x) == tuple and len(x) == 2 for x in box]):
            if box[0][1] == box[1][1] and box[2][1] == box[3][1] and box[0][1] >= 0 and box[3][1] <= image.shape[0]:
                if box[0][0] == box[2][0] and box[1][0] == box[3][0] and box[0][0] >= 0 and box[1][0] <= image.shape[1]:
                    return image[:box[3][1], :box[3][0], :][box[0][1]:, box[0][1]:, :]
        raise InvalidArgumentError(('Box',))


def shift_image(image, direction=(1, 1), fill_mode='fill', fill_value=0):
    non = lambda s: s if s < 0 else None
    pop = lambda s: s if s > 0 else None
    ox, oy = direction
    image = np.roll(image, oy, axis=0)
    image = np.roll(image, ox, axis=1)
    if fill_mode == 'fill':
        if 0 <= fill_value <= 255:
            image[non(oy):pop(oy), :, :] = fill_value
            image[:, non(ox):pop(ox), :] = fill_value
            return image
        raise InvalidArgumentError(('fill_value',))

    elif fill_mode == 'warp':
        return image
    else:
        raise InvalidMethodError(('Fill', 'Warp'))
