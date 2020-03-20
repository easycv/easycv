from cv.validators import Validator
from cv.errors.transforms import ArgumentNotProvidedError


class Transform:
    default_args = {}

    def __init__(self, **kwargs):
        for arg in self.default_args:
            if self.default_args[arg] is None and arg not in kwargs:
                raise ArgumentNotProvidedError(arg)
            elif isinstance(self.default_args[arg], Validator):
                validator = self.default_args[arg]
                kwargs[arg] = validator.check(arg, kwargs.get(arg))

        self._args = dict(self.default_args)
        self._args.update(kwargs)

    def args(self):
        return self._args

    def apply(self, image, **kwargs):
        pass

    def process(self, image):
        return self.apply(image, **self._args)

    def __call__(self, image):
        return self.process(image)

    def __str__(self):
        args_str = []
        for arg in self._args:
            value = (
                self._args[arg]
                if isinstance(self._args[arg], (str, int, float))
                else "(...)"
            )
            args_str.append("{}={}".format(arg, value))
        return "{} ({})".format(self.__class__.__name__, ", ".join(args_str))

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return isinstance(other, Transform) and self.args() == other.args()
