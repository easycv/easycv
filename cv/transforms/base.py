from cv.errors.transforms import ArgumentNotProvidedError


class Transform:
    default_args = {}

    def __init__(self, **kwargs):
        for arg in self.default_args:
            if self.default_args[arg] is None and arg not in kwargs:
                raise ArgumentNotProvidedError(arg)

        self.args = dict(self.default_args)
        self.args.update(kwargs)

    def apply(self, image, **kwargs):
        pass

    def __call__(self, image):
        return self.apply(image, **self.args)

    def __str__(self):
        args_str = []
        for arg in self.args:
            value = self.args[arg] if isinstance(self.args[arg], (str, int, float)) else '(...)'
            args_str.append(f'{arg}={value}')
        return f'{self.__class__.__name__} ({", ".join(args_str)})'

    def __repr__(self):
        return str(self)