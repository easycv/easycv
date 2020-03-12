from math import inf

import numpy as np

from cv.errors import ArgumentNotProvidedError, InvalidArgumentError


class Validator:
    def __init__(self, default=None):
        self._default = default

    def check(self, arg_name, arg):
        if arg is None:
            if self._default is None:
                raise ArgumentNotProvidedError(arg_name)
            else:
                return self._default
        else:
            return self.validate(arg_name, arg)

    def validate(self, arg_name, arg):
        pass


class Option(Validator):
    def __init__(self, options, default=None):
        self.options = options
        default = None if default is None else options[default]
        super().__init__(default=default)

    def validate(self, arg_name, arg):
        if arg not in self.options:
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Possible values: {", ".join(self.options)}')
        else:
            return arg


class Number(Validator):
    def __init__(self, min_value=-inf, max_value=inf, only_integer=False, only_odd=False, default=None):
        self._only_odd = only_odd
        self._min_value = min_value
        self._max_value = max_value
        self.only_integer = only_integer
        super().__init__(default=default)

    def validate(self, arg_name, arg, inside_list=False):
        allowed_types = (int,) if self.only_integer else (int, float)
        if not isinstance(arg, allowed_types) or not (self._min_value <= arg <= self._max_value) \
                or self._only_odd and arg % 2 == 0:
            if inside_list:
                prefix = "a list/tuple of " + ("integers" if self.only_integer else "numbers")
            else:
                if self.only_integer:
                    prefix = "an odd integer" if self._only_odd else "an integer"
                else:
                    prefix = "an odd number" if self._only_odd else "a number"
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be {prefix} '
                                       f'between {self._min_value} and {self._max_value}.')
        return arg


class Type(Validator):
    def __init__(self, arg_type, default=None):
        self.arg_type = arg_type
        super().__init__(default=default)

    def validate(self, arg_name, arg, inside_list=False):
        if not isinstance(arg, self.arg_type):
            prefix = "a list/tuple of objects" if inside_list else "an object"
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be {prefix} '
                                       f'from class {self.arg_type.__name__}')
        return arg


class List(Validator):
    def __init__(self, validator, length=None, default=None):
        self._validator = validator
        self._length = length
        super().__init__(default=default)

    def validate(self, arg_name, arg):
        if not isinstance(arg, (list, tuple, np.array)):
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be a list or tuple.')
        if self._length is not None and len(arg) != self._length:
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be {self._length} elements long.')

        for e in list(arg):
            self._validator.validate(arg_name, e, inside_list=True)
        return arg
