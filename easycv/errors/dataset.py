class InvalidClass(Exception):
    def __init__(self, key):
        super().__init__(
            "Class "+key+" doesn't exist in the dataset"
        )


class NoClassesGiven(Exception):
    def __init__(self):
        super().__init__(
            "A dataset must have at least on class"
        )
