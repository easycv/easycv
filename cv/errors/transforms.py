class ArgumentNotProvidedError(Exception):
    """Raised when a transform is called without a mandatory argument"""
    def __init__(self, argument):
        super().__init__(f'Must provide a value for argument: {argument}')


class InvalidArgumentError(Exception):
    """Raised when a transform is called with an invalid argument"""
    def __init__(self, argument):
        super().__init__(f'Invalid value for argument: {argument}')


class InvalidMethodError(Exception):
    """Raised when a transform is called with an invalid method"""
    def __init__(self, args):
        super().__init__(f'Invalid Method. Available methods: {", ".join(str(arg) for arg in args)}')
