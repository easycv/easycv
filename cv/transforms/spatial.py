from cv.errors.transforms import InvalidArgumentError


def crop_image(image, box=None):
    if type(box) == list and len(box) == 4 and all([type(x) == tuple and len(x)==2 for x in box]):
        if box[0][1] == box[1][1] and box[2][1] == box[3][1] and box[0][1] >= 0 and box[3][1] <= image.shape[0]:
            if box[0][0] == box[2][0] and box[1][0] == box[3][0] and box[0][0] >= 0 and box[1][0] <= image.shape[1]:
                return image[:box[3][1], :box[3][0], :][box[0][1]:, box[0][1]:, :]
    raise InvalidArgumentError(['Box'])

