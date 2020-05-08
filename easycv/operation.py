from easycv.errors import MissingArgumentError


class Operation:
    arguments = None  # Validators for arguments
    outputs = None  # Validators for outputs

    _args = None  # Argument values

    @property
    def args(self):
        """
        Returns a dictionary containing all names/values of the operation arguments. If the \
        operation isn't initialized yet this only returns the arguments provided when creating \
        the transform. Default arguments are only added on `initialize`.

        :return: Dictionary with all arguments and values.
        :rtype: :class:`dictionary`
        """
        return self._args

    def initialize(self, index=None, forwarded=()):
        """
        Initializes the operation. Initializing is verifying if the operation has all the \
        required arguments to run and adding the default values for the arguments that were not \
        provided or forwarded. If some required arguments are missing a \
        :class:`~easycv.errors.MissingArgumentError` is raised.

        :param index: Index of the operation, defaults to None
        :type index: :class:`int`, optional
        :param forwarded: List of arguments forwarded to the operation, defaults to no forwards
        :type forwarded: :class:`list`/:class:`tuple`, optional
        """

        for arg in self.arguments:
            if self.arguments[arg].default is None:
                if arg not in self._args and arg not in forwarded:
                    raise MissingArgumentError(arg, index=index)
            validator = self.arguments[arg]
            self._args[arg] = validator.default

    def can_be_forwarded(self, arg_name, validator):
        """
        Checks if the argument can be forwarded into this operation. An argument can be forwarded \
        if and only if it is one of the operation arguments, wasn't specified manually and the \
        validators are compatible.

        :param arg_name: Name of the argument to check
        :type arg_name: :class:`str`
        :param validator: Validator of the argument to check
        :type validator: :class:`~easycv.validators.validator`
        :return: *True* if the argument can be forwarded into the operation *False* otherwise
        :rtype: :class:`bool`
        """

        if arg_name in self.arguments and arg_name not in self._args:
            return self.arguments[arg_name].accepts(validator)
        return False
