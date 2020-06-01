import sys
from functools import wraps
from types import FunctionType

from easycv.transforms.noise import Noise
from easycv.transforms.filter import Blur, Sharpness, Sharpen
from easycv.transforms.perspective import Perspective
from easycv.transforms.edges import Gradient, GradientAngle, Canny, Lines, Circles
from easycv.transforms.color import (
    GammaCorrection,
    GrayScale,
    FilterChannels,
    PhotoSketch,
    Negative,
    Cartoon,
)
from easycv.transforms.spatial import Resize, Crop, Rotate, Translate
from easycv.transforms.selectors import Select
from easycv.transforms.detect import Scan

transforms = [
    Blur,
    Canny,
    Circles,
    Cartoon,
    Crop,
    FilterChannels,
    GammaCorrection,
    Gradient,
    GradientAngle,
    GrayScale,
    Lines,
    Negative,
    Noise,
    Perspective,
    PhotoSketch,
    Resize,
    Rotate,
    Scan,
    Select,
    Sharpen,
    Sharpness,
    Translate,
]

__all__ = [transform.__name__ for transform in transforms]


def show_args(function, exclude_method=False):
    @wraps(function)
    def inner(transform, **kwargs):
        if exclude_method and kwargs.get("method") is not None:
            kwargs.pop("method")
        return function(transform, arguments=kwargs)

    return inner


def create_function(name, first_arg, args, defaults, function_code):
    formatted_args = ", ".join("{}".format(arg) for arg in args)
    formatted_args = (
        formatted_args + ", **kwargs" if args else formatted_args + " **kwargs"
    )
    function_code = "def {}({}, {}): {}".format(
        name, first_arg, formatted_args, function_code
    )
    compiled_func = compile(function_code, name, "exec")
    function = FunctionType(compiled_func.co_consts[0], globals(), name)
    function.__defaults__ = tuple(defaults)
    return function


def add_method_function(transform, method_name, default_values):
    code = 'return cls(method="{}", **kwargs["arguments"])'.format(method_name)
    method = create_function(
        method_name, "cls", default_values, tuple(default_values.values()), code
    )
    method.__doc__ = transform.__doc__
    setattr(transform, method_name, classmethod(show_args(method, exclude_method=True)))


if "sphinx" not in sys.modules:
    for transform in transforms:
        default_values = transform.get_default_values()
        code = "super(self.__class__, self).__init__(**kwargs['arguments'])"
        init = create_function(
            "temp", "self", default_values, tuple(default_values.values()), code
        )
        transform.__init__ = show_args(init)

        for method in transform.get_methods():
            default_values = transform.get_default_values(method=method)
            add_method_function(transform, method, default_values)
