class Transform:
    arguments = {}

    def __init__(self, **kwargs):
        for arg in self.arguments:
            if self.arguments[arg] is None and arg not in kwargs:
                raise ValueError('')

        self.arguments.update(kwargs)

    def apply(self, image):
        pass

    def __call__(self, image):
        return self.apply(image)
