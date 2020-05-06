class ArgumentNotProvidedError(Exception):
    """Raised when a transform is called without a mandatory argument"""

    def __init__(self, argument):
        super().__init__("Must provide a value for argument: {}".format(argument))


class InvalidArgumentError(Exception):
    """Raised when a transform is called with an invalid argument"""

    def __init__(self, message):
        super().__init__(message)


class ValidatorError(Exception):
    def __init__(self, desc):
        self.desc = desc

    def get_description(self):
        return self.desc


class InvalidMethodError(Exception):
    """Raised when a transform is called with an invalid method"""

    def __init__(self, args):
        super().__init__(
            "Invalid Method. Available methods: {}".format(
                ", ".join(str(arg) for arg in args)
            )
        )


class UnsupportedArgumentError(Exception):
    """Raised when a transform is created with an unsupported argument"""

    def __init__(self, transform, method=None):
        if method is None:
            name = 'transform "{}"'.format(transform.__class__.__name__)
        else:
            name = 'method "{}"'.format(method)
        msg = "Invalid arguments for {}. ".format(name)
        if transform.arguments:
            msg += "Allowed arguments: {}".format(", ".join(transform.arguments))
        else:
            msg += "{} takes no arguments.".format(name.title())
        super().__init__(msg)


class MissingArgumentError(Exception):
    """Raised when an transform is executed while missing a mandatory argument"""

    def __init__(self, arg, index=None):
        if index is None:
            msg = "Transform "
        else:
            msg = "Transform at index {} ".format(index)
        super().__init__(msg + "missing mandatory argument: {}".format(arg))


class InvalidSelectionError(Exception):
    """Raised when an invalid selection is made in a selector"""

    def __init__(self, msg):
        super().__init__("Invalid selection. {}".format(msg))
