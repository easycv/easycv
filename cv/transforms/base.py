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

    def __str__(self):
        args_str = []
        for arg in self.arguments:
            value = self.arguments[arg] if isinstance(self.arguments[arg], (str, int)) else '(...)'
            args_str.append(f'{arg}={value}')
        return f'{self.__class__.__name__} ({", ".join(args_str)})'

    def __repr__(self):
        return str(self)