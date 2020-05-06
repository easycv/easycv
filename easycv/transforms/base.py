from easycv.errors import (
    UnsupportedArgumentError,
    InvalidMethodError,
    ArgumentNotProvidedError,
    MissingArgumentError,
)

import cv2
from copy import copy


class Metadata(type):
    exclude = {"apply", "default_args"}

    def __dir__(cls):
        return list(set(cls.__dict__.keys()) - cls.exclude)


class Transform(metaclass=Metadata):
    methods = None
    default_method = None
    method_name = "method"

    arguments = None
    outputs = None

    def __init__(self, **kwargs):
        self._method = self._extract_method(kwargs)
        self.arguments = self._extract_attribute("arguments")
        self.outputs = self._extract_attribute("outputs")

        if any(arg not in self.arguments for arg in kwargs):
            raise UnsupportedArgumentError(self, method=self._method)

        self._args = {"method": self._method}
        for arg in kwargs:
            validator = self.arguments[arg]
            self._args[arg] = validator.check(arg, kwargs)

        self._unspecified = [arg for arg in self.arguments if arg not in kwargs]
        self._required = [
            arg for arg in self._unspecified if self.arguments[arg].required
        ]

    def initialize(self, index=None, forwarded=()):
        for arg in self._unspecified:
            if arg in self._required and arg not in forwarded:
                raise MissingArgumentError(arg, index=index)
            validator = self.arguments[arg]
            self._args[arg] = validator.default

    @property
    def required(self):
        return self._required

    def can_be_forwarded(self, name, validator):
        if name in self._unspecified:
            return self.arguments[name].accept(validator)
        return False

    def _extract_method(self, kwargs):
        if self.contains_methods:
            method = self.default_method
            if "method" in kwargs:
                method = kwargs.pop("method")
                if method not in self.methods:
                    raise InvalidMethodError(tuple(self.methods))
            if method is None:
                raise ArgumentNotProvidedError("method")
            return method
        else:
            return None

    def _extract_attribute(self, name):
        collection = self.arguments if name == "arguments" else self.outputs
        if collection is not None:
            to_keep = collection.keys()
            if self.contains_methods and isinstance(self.methods, dict):
                if self.methods[self._method] is None:
                    to_keep = []
                elif len(self.methods[self._method]) > 0:
                    args = self.methods[self._method].get(name)
                    if args is not None:
                        to_keep = args
            return {key: collection[key] for key in to_keep}
        else:
            return {}

    @property
    def contains_methods(self):
        return self.methods is not None

    @property
    def contains_outputs(self):
        return self.outputs is not None

    def __call__(self, image, forwarded=None):
        output = self.process(image, forwarded=forwarded)

        if isinstance(output, dict):
            return output
        else:
            if output.min() >= 0 and output.max() <= 255:
                if output.dtype.kind != "i":
                    if output.min() >= 0 and output.max() <= 1:
                        output = output * 255
                    output = output.astype("uint8")
            else:
                output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(
                    "uint8"
                )
            return {"image": output}

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

    def copy(self):
        return copy(self)

    def _get_arguments(self, kwargs, methods):
        arguments = {}
        if methods is not None:
            self.method_name = methods.method_name
            method = methods.check("method", kwargs)
            arguments[self.method_name] = method

            if methods.contains_allowed:
                valid_arguments = methods.allowed_args(method)
            else:
                valid_arguments = list(self.arguments.keys())
                valid_arguments.remove("method")
        else:
            valid_arguments = self.arguments.keys()

        for arg in valid_arguments:
            validator = self.arguments[arg]
            arguments[arg] = validator.check(arg, kwargs)

        return arguments

    @property
    def default_values(self):
        return {
            arg_name: self.arguments[arg_name].default for arg_name in self.arguments
        }

    @classmethod
    def get_default_values(cls, method=None):
        default_values = {}
        if method is not None:
            method_validator = cls.arguments["method"]
            if method_validator.contains_allowed:
                arguments = method_validator.allowed_args(method)
            else:
                arguments = cls.arguments
        else:
            arguments = cls.arguments

        if cls.arguments is not None:
            for argument in arguments:
                if method is None or argument != "method":
                    default_values[argument] = cls.arguments[argument].default

        return default_values

    @classmethod
    def get_methods(cls):
        method_validator = cls.arguments.get("method")
        if method_validator is not None:
            return method_validator.allowed_methods
        return []

    def apply(self, image, **kwargs):
        return image

    def args(self):
        return self._args

    def process(self, image, forwarded=None):
        if forwarded is None:
            args = self._args
        else:
            args = self._args.copy()
            args.update(forwarded)

        return self.apply(image, **args)
