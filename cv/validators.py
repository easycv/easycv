import numpy as np

from cv.errors import ArgumentNotProvidedError, InvalidArgumentError


class Validator:
    def __init__(self, default=None):
        self._default = default

    def check(self, arg_name, kwargs):
        arg = kwargs.get(arg_name)
        if arg is None:
            if self._default is None:
                raise ArgumentNotProvidedError(arg_name)
            else:
                return self._default
        else:
            return self.validate(arg_name, kwargs)

    def validate(self, arg_name, kwargs, inside_list=False):
        pass


class Option(Validator):
    def __init__(self, options, default=None):
        self.options = options
        default = None if default is None else options[default]
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.get(arg_name)
        if arg not in self.options:
            raise InvalidArgumentError(
                'Invalid value for "{}". '.format(arg_name)
                + "Possible values: {}".format(", ".join(self.options))
            )
        else:
            return arg


class Number(Validator):

    def __init__(
            self,
            min_value=-float("inf"),
            max_value=float("inf"),
            only_integer=False,
            only_odd=False,
            default=None,
    ):
        self._only_odd = only_odd
        self._min_value = min_value
        self._max_value = max_value
        self.only_integer = only_integer
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.get(arg_name)
        allowed_types = (int,) if self.only_integer else (int, float)
        if (
                not isinstance(arg, allowed_types)
                or not (self._min_value <= arg <= self._max_value)
                or self._only_odd
                and arg % 2 == 0
        ):
            if inside_list:
                prefix = "a list/tuple of " + (
                    "integers" if self.only_integer else "numbers"
                )
            else:
                if self.only_integer:
                    prefix = "an odd integer" if self._only_odd else "an integer"
                else:
                    prefix = "an odd number" if self._only_odd else "a number"
            raise InvalidArgumentError(
                'Invalid value for "{}". Must be {} '.format(arg_name, prefix)
                + "between {} and {}.".format(self._min_value, self._max_value)
            )
        return arg


class Method(Validator):
    def __init__(self, options, default=None):
        self.options = options
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.pop(arg_name)
        if arg not in self.options:
            raise InvalidArgumentError(
                "Invalid method. Available methods: {}".format(", ".join(self.options))
            )
        if any(a not in self.options[arg] for a in kwargs):
            raise InvalidArgumentError(
                'Invalid arguments for method "{}". '.format(arg)
                + "Allowed arguments: {}".format(", ".join(self.options[arg]))
            )
        else:
            return arg


class Type(Validator):
    def __init__(self, arg_type, default=None):
        self.arg_type = arg_type
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.get(arg_name)
        if not isinstance(arg, self.arg_type):
            prefix = "a list/tuple of objects" if inside_list else "an object"
            raise InvalidArgumentError(
                'Invalid value for "{}". '.format(arg_name)
                + "Must be {} from class {}".format(prefix, self.arg_type.__name__)
            )
        return arg


class List(Validator):
    def __init__(self, validator, length=None, default=None):
        self._validator = validator
        self._length = length
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.get(arg_name)
        if not isinstance(arg, (list, tuple, np.array)):
            raise InvalidArgumentError(
                'Invalid value for "{}". Must be a list or tuple.'.format(arg_name)
            )
        if self._length is not None and len(arg) != self._length:
            raise InvalidArgumentError(
                'Invalid value for "{}". Must be {} elements long.'.format(
                    arg_name, self._length
                )
            )

        for e in list(arg):
            self._validator.validate(arg_name, {arg_name: e}, inside_list=True)
        return arg
