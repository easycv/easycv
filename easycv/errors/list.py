class InvalidListInputSource(Exception):
    def __init__(self):
        super().__init__("Lists can only be created from a list of images.")
