import cv2


def nearest_square_side(n):
    i = 1
    while i ** 2 < n:
        i += 1
    return i


interpolation_methods = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
