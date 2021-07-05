class InvalidClassError(Exception):
    def __init__(self, key):
        super().__init__(
            "Class "+key+" doesn't exist in the dataset"
        )


class NoClassesGivenError(Exception):
    def __init__(self):
        super().__init__(
            "A dataset must have at least one class"
        )
