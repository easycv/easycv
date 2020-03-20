class ArgumentNotProvidedError(Exception):
    """Raised when a transform is called without a mandatory argument"""

    def __init__(self, argument):
        super().__init__("Must provide a value for argument: {}".format(argument))


class InvalidArgumentError(Exception):
    """Raised when a transform is called with an invalid argument"""

    def __init__(self, message):
        super().__init__(message)


class InvalidMethodError(Exception):
    """Raised when a transform is called with an invalid method"""

    def __init__(self, args):
        super().__init__(
            "Invalid Method. Available methods: {}".format(
                ", ".join(str(arg) for arg in args)
            )
        )
