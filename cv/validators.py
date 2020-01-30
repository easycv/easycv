from math import inf
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
    def __init__(self, min_value=-inf, max_value=inf, only_integer=False, default=None):
        self._min_value = min_value
        self._max_value = max_value
        self.only_integer = only_integer
        super().__init__(default=default)

    def validate(self, arg_name, arg, inside_list=True):
        allowed_types = (int,) if self.only_integer else (int, float)
        if not isinstance(arg, allowed_types) or not (self._min_value <= arg <= self._max_value):
            if inside_list:
                prefix = "a list/tuple of " + ("integers" if self.only_integer else "numbers")
            else:
                prefix = "an integer" if self.only_integer else "a number"
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be {prefix} between {self._min_value} and {self._max_value}.')
        return arg


class List(Validator):
    def __init__(self, validator, default=None):
        self._validator = validator
        super().__init__(default=default)

    def validate(self, arg_name, arg):
        if not isinstance(arg, (list, tuple)):
            raise InvalidArgumentError(f'Invalid value for "{arg_name}". Must be a list or tuple.')
        for e in arg:
            self._validator.validate(arg_name, e, inside_list=True)
        return arg
