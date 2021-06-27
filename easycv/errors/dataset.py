class InvalidClass(Exception):
    def __init__(self, key):
        super().__init__(
            "Class "+key+" doesn't exist in the dataset"
        )

