from easycv.transforms.base import Transform


class Crop(Transform):
    default_args = {"box": []}

    def apply(self, image, **kwargs):
        pass


class Shift(Transform):
    default_args = {"direction": (1, 1), "fill_mode": "fill", "fill_value": 0}

    def apply(self, image, **kwargs):
        pass
