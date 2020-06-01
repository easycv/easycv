import cv2

from easycv.operation import Operation
from easycv.errors import (
    UnsupportedArgumentError,
    InvalidMethodError,
    ArgumentNotProvidedError,
)


class Metadata(type):
    exclude = {
        "run",
        "process",
        "arguments",
        "outputs",
        "method_name",
        "methods",
        "default_method",
    }

    def __dir__(cls):
        return list(set(cls.__dict__.keys()) - cls.exclude)


class Transform(Operation, metaclass=Metadata):
    methods = None
    default_method = None
    method_name = "method"

    def __init__(self, **kwargs):
        self._method = self._extract_method(kwargs)
        self.arguments = self._extract_attribute("arguments", self._method)
        self.outputs = self._extract_attribute("outputs", self._method)

        if any(arg not in self.arguments for arg in kwargs):
            raise UnsupportedArgumentError(self, method=self._method)

        if self._method is not None:
            self._args = {self.method_name: self._method}
        else:
            self._args = {}

        for arg in kwargs:
            validator = self.arguments[arg]
            validator.check(arg, kwargs[arg])
            self._args[arg] = kwargs[arg]

    def __call__(self, image, forwarded=None):
        output = self.run(image, forwarded=forwarded)

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
        return isinstance(other, Transform) and self.args == other.args

    def __repr__(self):
        self.initialize()
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

    @property
    def contains_outputs(self):
        return self.outputs is not None

    @classmethod
    def contains_methods(cls):
        return cls.methods is not None

    @classmethod
    def get_methods(cls):
        return list(cls.methods) if cls.methods is not None else []

    @classmethod
    def get_default_values(cls, method=None):
        if method is None:
            if cls.default_method is not None:
                default_values = {"method": cls.default_method}
                args = cls._extract_attribute("arguments", cls.default_method)
            else:
                default_values = {}
                args = cls.arguments
        else:
            default_values = {}
            args = cls._extract_attribute("arguments", method)

        if cls.arguments is not None:
            for argument in args:
                default_values[argument] = args[argument].default

        return default_values

    @classmethod
    def _extract_attribute(cls, name, method):
        collection = cls.arguments if name == "arguments" else cls.outputs
        if collection is not None:
            to_keep = collection.keys()
            if cls.contains_methods() and isinstance(cls.methods, dict):
                if cls.methods[method] is None:
                    to_keep = []
                elif len(cls.methods[method]) > 0:
                    args = cls.methods[method].get(name)
                    if args is not None:
                        to_keep = args
            return {key: collection[key] for key in to_keep}
        else:
            return {}

    def _extract_method(self, kwargs):
        if self.contains_methods():
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

    def process(self, image, **kwargs):
        pass

    def run(self, image, forwarded=None):
        self.initialize()
        if forwarded is None:
            args = self._args
        else:
            args = self._args.copy()
            args.update(forwarded)

        return self.process(image, **args)
