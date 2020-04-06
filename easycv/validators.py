import numpy as np

from easycv.errors import ArgumentNotProvidedError, InvalidArgumentError


class Validator:
    """
    This class is the base for all Validators. All Validators should be child from this class \
    and override the `validate` method.
    If default is set to None the argument is required and the Validator will throw an \
    :class:`~easycv.errors.transforms.ArgumentNotProvidedError`.

    :param default: Default value for the argument, defaults to None
    :type default: :class:`object`, optional
    """

    def __init__(self, default=None):
        self._default = default

    def check(self, arg_name, kwargs):
        """
        Check if an argument satisfies the Validator conditions. If the default value is None
        the Validator throws an :class:`~easycv.errors.transforms.ArgumentNotProvidedError`, \
        otherwise it calls the validate method to make sure all conditions are verified and if \
        not, throws an :class:`~easycv.errors.transforms.InvalidArgumentError` with the \
        appropriate error message.

        :param arg_name: Name of the argument to validate
        :type arg_name: :class:`str`
        :param kwargs: Dictionary containing all arguments and their values
        :type kwargs: :class:`dict`
        """
        arg = kwargs.get(arg_name)
        if arg is None:
            if self._default is None:
                raise ArgumentNotProvidedError(arg_name)
            else:
                return self._default
        else:
            return self.validate(arg_name, kwargs)

    def validate(self, arg_name, kwargs, inside_list=False):
        """
        Every Validator should override this method. This method should validate the arguments \
        named `arg_name` from `kwargs` by checking if it verifies the constrains. If the argument \
        is invalid it should throw an :class:`~easycv.errors.transforms.InvalidArgumentError` \
        with a message clarifying what is invalid and how to correct it. If this Validator is \
        inside a the message should reflect that.

        :param arg_name: Name of the argument to validate
        :type arg_name: :class:`str`
        :param kwargs: Dictionary containing all arguments and their values
        :type kwargs: :class:`dict`
        :param inside_list: True if this Validator is inside a List, defaults to False
        :type inside_list: :class:`bool`, optional
        """
        pass


class Number(Validator):
    """
    Validator to check if an argument is a number. More restrictions can be applied through \
    the keyword arguments.

    :param min_value: Minimum value allowed, defaults to -inf
    :type min_value: :class:`int`/:class:`float`, optional
    :param max_value: Maximum value allowed, defaults to inf
    :type max_value: :class:`int`/:class:`float`, optional
    :param only_integer: Allow only integers, defaults to False
    :type only_integer: :class:`bool`, optional
    :param only_odd: Allow only odd numbers, defaults to False
    :type only_odd: :class:`bool`, optional
    """

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


class Option(Validator):
    """
    Validator to check if an argument one of the allowed options.

    :param options: Allowed options
    :type options: :class:`list`
    """

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


class Method(Validator):
    """
    Validator to check if a method is valid and if the arguments called for that method are valid.

    :param methods: Dictionary with methods as key and values are lists with valid args for the \
    corresponding method
    :type methods: :class:`dict`
    """

    def __init__(self, methods, default=None):
        self.methods = methods
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.pop(arg_name)
        if arg not in self.methods:
            raise InvalidArgumentError(
                "Invalid method. Available methods: {}".format(", ".join(self.methods))
            )
        if any(a not in self.methods[arg] for a in kwargs):
            raise InvalidArgumentError(
                'Invalid arguments for method "{}". '.format(arg)
                + "Allowed arguments: {}".format(", ".join(self.methods[arg]))
            )
        else:
            return arg


class Type(Validator):
    """
    Validator to check if an argument is from the specified type.

    :param arg_type: Allowed type
    :type arg_type: :class:`type`
    """

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
    """
    Validator to check if an argument is a list/tupple or a numpy array containing only elements \
    that satisfies the given Validator.

    :param validator: Validator to apply to each element
    :type validator: :class:`Validator`
    :param length: Mandatory length, defaults to None
    :type length: :class:`int`, optional
    """

    def __init__(self, validator, length=None, default=None):
        self._validator = validator
        self._length = length
        super().__init__(default=default)

    def validate(self, arg_name, kwargs, inside_list=False):
        arg = kwargs.get(arg_name)
        if not isinstance(arg, (list, tuple, np.ndarray)):
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
