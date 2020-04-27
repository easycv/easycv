from easycv.errors import InvalidArgumentError

import cv2


class Metadata(type):
    exclude = {"apply", "default_args"}

    def __dir__(cls):
        return list(set(cls.__dict__.keys()) - cls.exclude)


class Transform(metaclass=Metadata):
    inputs = {}
    outputs = {}

    method_name = None

    def __init__(self, **kwargs):
        methods = self.inputs.get("method")
        if methods is not None and methods.contains_allowed:
            methods.add_unspecified_allowed_args(self.inputs.keys())
        elif any(arg not in self.inputs for arg in kwargs):
            msg = 'Invalid arguments for transform "{}". '.format(
                self.__class__.__name__
            )
            if self.inputs:
                msg += "Allowed arguments: {}".format(", ".join(self.inputs))
            else:
                msg += "{} takes no arguments.".format(self.__class__.__name__)
            raise InvalidArgumentError(msg)
        self._args = self._get_arguments(kwargs, methods)

    def __call__(self, image):
        output = self.process(image)

        if output.min() >= 0 and output.max() <= 255:
            if output.dtype.kind != "i":
                if output.min() >= 0 and output.max() <= 1:
                    output = output * 255
                output = output.astype("uint8")
        else:
            output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(
                "uint8"
            )

        return output

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
                valid_arguments = list(self.inputs.keys())
                valid_arguments.remove("method")
        else:
            valid_arguments = self.inputs.keys()

        for arg in valid_arguments:
            validator = self.inputs[arg]
            arguments[arg] = validator.check(arg, kwargs)

        return arguments

    @property
    def default_values(self):
        return {arg_name: self.inputs[arg_name].default for arg_name in self.inputs}

    @classmethod
    def add_unspecified_allowed_args(cls):
        method_validator = cls.inputs.get("method")
        if method_validator is not None and method_validator.contains_allowed:
            method_validator.add_unspecified_allowed_args(cls.inputs)

    @classmethod
    def get_default_values(cls, method=None):
        default_values = {}
        if method is not None:
            method_validator = cls.inputs["method"]
            if method_validator.contains_allowed:
                arguments = method_validator.allowed_args(method)
            else:
                arguments = cls.inputs
        else:
            arguments = cls.inputs

        for argument in arguments:
            if method is None or argument != "method":
                default_values[argument] = cls.inputs[argument].default

        return default_values

    @classmethod
    def get_methods(cls):
        method_validator = cls.inputs.get("method")
        if method_validator is not None:
            return method_validator.allowed_methods
        return []

    def apply(self, image, **kwargs):
        return image

    def args(self):
        return self._args

    def process(self, image):
        return self.apply(image, **self._args)
