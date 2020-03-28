class InvalidPathError(Exception):
    """Raised when a path is invalid"""

    pass


class ImageDownloadError(Exception):
    """Raised when downloading the image fails"""

    pass


class ImageDecodeError(Exception):
    """Raised when decoding the image fails"""

    pass


class ImageSaveError(Exception):
    """Raised when saving the image fails"""

    pass


class InvalidPipelineInputSource(Exception):
    def __init__(self):
        super().__init__(
            "Pipelines can only be created from a list of transforms/pipelines or a saved pipeline"
        )


class InvalidImageInputSource(Exception):
    def __init__(self):
        super().__init__(
            "Images can only be created from a path/url or a numpy array representing the image"
        )


class ImageNotLoaded(Exception):
    def __init__(self):
        super().__init__(
            "Images can only be created from a path/url or a numpy array representing the image"
        )
