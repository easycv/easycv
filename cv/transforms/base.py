class Transform:
    arguments = {}
    mandatory = []

    def __init__(self, **kwargs):
        for mandatory_arg in self.mandatory:
            if mandatory_arg not in kwargs:
                raise ValueError('')

        self.arguments.update(kwargs)

    def apply(self, image):
        pass

    def __call__(self, image):
        return self.apply(image)
