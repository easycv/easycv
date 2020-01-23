from cv.errors.transforms import ArgumentNotProvidedError


class Transform:
    arguments = {}

    def __init__(self, **kwargs):
        for arg in self.arguments:
            if self.arguments[arg] is None and arg not in kwargs:
                raise ArgumentNotProvidedError(arg)

        self.arguments.update(kwargs)

    def apply(self, image):
        pass

    def __call__(self, image):
        return self.apply(image)
