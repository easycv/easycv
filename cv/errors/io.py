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