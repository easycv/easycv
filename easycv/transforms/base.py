from easycv.errors import InvalidArgumentError


class Metadata(type):
    exclude = {"apply", "default_args"}

    def __dir__(cls):
        return list(set(cls.__dict__.keys()) - cls.exclude)


class Transform(metaclass=Metadata):
    default_args = {}
    method_name = None

    def __init__(self, **kwargs):
        methods = self.default_args.get("method")
        if methods is not None and methods.contains_allowed:
            methods.add_unspecified_allowed_args(self.default_args.keys())
        elif any(arg not in self.default_args for arg in kwargs):
            msg = 'Invalid arguments for transform "{}". '.format(
                self.__class__.__name__
            )
            if self.default_args:
                msg += "Allowed arguments: {}".format(", ".join(self.default_args))
            else:
                msg += "{} takes no arguments.".format(self.__class__.__name__)
            raise InvalidArgumentError(msg)
        self._args = self._get_arguments(kwargs, methods)

    def __call__(self, image):
        return self.process(image)

    def __eq__(self, other):
        return isinstance(other, Transform) and self.args() == other.args()

    def __repr__(self):
        return str(self)

    def __str__(self):
        args_str = []
        for arg in self._args:
            value = (
                self._args[arg]
                if isinstance(self._args[arg], (str, int, float))
                else "(...)"
            )
            if arg == self.method_name:
                arg = "method"
            args_str.append("{}={}".format(arg, value))
        return "{} ({})".format(self.__class__.__name__, ", ".join(args_str))

    def _get_arguments(self, kwargs, methods):
        arguments = {}
        if methods is not None:
            self.method_name = methods.method_name
            method = methods.check("method", kwargs)
            arguments[self.method_name] = method

            if methods.contains_allowed:
                valid_arguments = methods.allowed_args(method)
            else:
                valid_arguments = list(self.default_args.keys())
                valid_arguments.remove("method")
        else:
            valid_arguments = self.default_args.keys()

        for arg in valid_arguments:
            validator = self.default_args[arg]
            arguments[arg] = validator.check(arg, kwargs)

        return arguments

    @property
    def default_values(self):
        return {
            arg_name: self.default_args[arg_name].default
            for arg_name in self.default_args
        }

    @classmethod
    def add_unspecified_allowed_args(cls):
        method_validator = cls.default_args.get("method")
        if method_validator is not None and method_validator.contains_allowed:
            method_validator.add_unspecified_allowed_args(cls.default_args)

    @classmethod
    def get_default_values(cls, method=None):
        default_values = {}
        if method is not None:
            method_validator = cls.default_args["method"]
            if method_validator.contains_allowed:
                arguments = method_validator.allowed_args(method)
            else:
                arguments = cls.default_args
        else:
            arguments = cls.default_args

        for argument in arguments:
            default_values[argument] = cls.default_args[argument].default

        return default_values

    @classmethod
    def get_methods(cls):
        method_validator = cls.default_args.get("method")
        if method_validator is not None:
            return method_validator.allowed_methods
        return []

    def apply(self, image, **kwargs):
        pass

    def args(self):
        return self._args

    def process(self, image):
        return self.apply(image, **self._args)
