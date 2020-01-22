class InvalidArgsError(Exception):
    """Raised when a path is invalid"""
    def __init__(self, args):
        self.value = f'Invalid Argument(s): {" ".join(str(arg) for arg in args)}'

    def __str__(self):
        return repr(self.value)


class InvalidMethodError(Exception):
    """Raised when a path is invalid"""
    def __init__(self, args):
        self.value = f'Invalid Method. Available methods: {" ,".join(str(arg) for arg in args)}'

    def __str__(self):
        return repr(self.value)
