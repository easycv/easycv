class InvalidArgumentError(Exception):
    """Raised when a transform is called with invalid arguments"""
    def __init__(self, args):
        super().__init__(f'Invalid Argument(s): {", ".join(str(arg) for arg in args)}')


class InvalidMethodError(Exception):
    """Raised when a transform is called with an invalid method"""
    def __init__(self, args):
        super().__init__(f'Invalid Method. Available methods: {", ".join(str(arg) for arg in args)}')
