from easycv.validators import Method


class Transform:
    default_args = {}
    methods = None

    def __init__(self, **kwargs):
        if self.methods is not None:
            self._add_unspecified_args()
        self._args = self._get_arguments(kwargs)

    def _add_unspecified_args(self):
        specified_args = set(sum(self.methods.values(), []))
        not_specified = set(self.default_args) - specified_args - {"method"}
        if not_specified:
            for method in self.methods:
                self.methods[method].extend(not_specified)

    def _get_arguments(self, kwargs):
        if self.methods is not None:
            validator = Method(self.methods, default=self.default_args.get("method"))
            method = validator.check("method", kwargs)
            kwargs["method"] = method
            valid_arguments = self.methods[method]
        else:
            valid_arguments = self.default_args

        for arg in valid_arguments:
            validator = self.default_args[arg]
            kwargs[arg] = validator.check(arg, kwargs)

        return kwargs

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
